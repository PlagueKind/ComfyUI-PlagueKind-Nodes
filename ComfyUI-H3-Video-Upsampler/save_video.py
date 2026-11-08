# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""HTC Save Video -- frames in, an MP4 out, without the long-form RAM bill.

The stock tail of a chain graph is `RTX Video Frame Generation -> CreateVideo ->
SaveVideo`. On a 2.5-minute film that is where the machine falls over, and the
render is already done by then:

1. frame generation allocates its whole output at once, as float32: twice the
   frames at 12 bytes a pixel (a 3,600-frame film becomes a ~90 GB tensor),
   while the chain's own frames stay cached beside it;
2. the save then walks that tensor one frame at a time in Python, scales it to
   8 bits, converts RGB to YUV on one CPU core, and hands it to libx264 -- at
   crf 10, the slowest, biggest setting there is.

This node does both jobs in one pass and never holds more than a few frames:

- frames go up to the GPU in small chunks; frame generation (optional, the same
  NVIDIA Video Effects library the RTX node uses) runs on each pair there, and
  every frame -- source or generated -- is converted to 8-bit NV12 on the GPU,
  so only 1.5 bytes a pixel ever come back to RAM;
- NV12 is the hardware encoder's native input, so NVENC encodes it with no CPU
  colour conversion at all; a second thread muxes while the GPU converts the
  next chunk;
- quality is constant-quality (NVENC `cq`, x264 `crf`), not a fixed bitrate, so
  a quiet dialogue hop and a rain-soaked chase each get what they need;
- a keyframe is forced on every hard cut (a thumbnail difference, measured on
  the GPU), and at least every two seconds, so editors scrub the film cleanly.
  The encoders' own scene detection is off: on grainy footage it fires on
  ordinary frames;
- the soundtrack is trimmed or padded to the picture and interleaved as it goes.

Without CUDA the same path runs on the CPU with libx264/libx265/SVT-AV1.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from fractions import Fraction

import av
import numpy as np
import torch
import torch.nn.functional as F

TAG = "H3SaveVideo"

# Constant-quality targets per codec family. NVENC cq and x264 crf are not the
# same scale; these pairs were picked to look alike. "max" is the default: no
# visible loss on H3 output at a fraction of crf 10's size.
QUALITY = {
    "high":    {"cq": 21, "crf": 18, "audio_kbps": 192},
    "max":     {"cq": 17, "crf": 14, "audio_kbps": 256},
    "archive": {"cq": 12, "crf": 10, "audio_kbps": 320},
}
GPU_CODECS = {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "av1": "av1_nvenc"}
CPU_CODECS = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}
FRAME_GEN = {"off": 1, "x2": 2, "x3": 3, "x4": 4}
FRAME_GEN_MODE = "MEDIUM"   # the RTX node's default
CHUNK_PIXELS = 24 * 1024 * 1024   # frames converted per GPU round trip, by area
QUEUE_CHUNKS = 3            # converted chunks waiting for the muxer (bounds RAM)
CUT_THRESHOLD = 0.12        # mean |dY| over a 16x16 thumbnail, 0..1, that marks a cut
KEYFRAME_SECONDS = 2.0      # longest stretch between keyframes


def rgb_to_nv12(rgb):
    """[B, H, W, 3] float RGB (0..1, even H and W) -> ([B, H*3/2, W] uint8 NV12, [B, H, W] float Y).

    BT.709 matrix, limited range, chroma averaged over each 2x2 block. Runs on
    whatever device `rgb` is on.
    """
    x = rgb.float().clamp_(0.0, 1.0)
    r, g, b = x.unbind(-1)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    bsz, h, w = y.shape
    cb = F.avg_pool2d(cb.unsqueeze(1), 2).squeeze(1)
    cr = F.avg_pool2d(cr.unsqueeze(1), 2).squeeze(1)
    y8 = (16.0 + 219.0 * y).round_().clamp_(0, 255).to(torch.uint8)
    uv = torch.stack((128.0 + 224.0 * cb, 128.0 + 224.0 * cr), dim=-1)
    uv8 = uv.round_().clamp_(0, 255).to(torch.uint8).reshape(bsz, h // 2, w)
    return torch.cat((y8, uv8), dim=1), y


def _pick_codec(codec, use_gpu, width, height):
    family = "h264" if codec == "auto" else codec
    if use_gpu and width >= 160 and height >= 160:
        name = GPU_CODECS[family]
        if name in av.codecs_available:
            return name
    return CPU_CODECS[family]


def _encoder_options(name, quality, gop):
    q = QUALITY[quality]
    if name.endswith("_nvenc"):
        opts = {"preset": "p5", "tune": "hq", "rc": "vbr", "cq": str(q["cq"]),
                "b": "0", "spatial-aq": "1", "rc-lookahead": "20", "forced-idr": "1",
                "g": str(gop), "bf": "0", "no-scenecut": "1"}
        if not name.startswith("av1"):
            opts["temporal-aq"] = "1"
        return opts
    if name == "libx264":
        return {"crf": str(q["crf"]), "preset": "medium", "forced-idr": "1",
                "x264-params": f"keyint={gop}:min-keyint=1:scenecut=0:bframes=0"}
    if name == "libx265":
        return {"crf": str(q["crf"]), "preset": "medium", "forced-idr": "1",
                "x265-params": f"keyint={gop}:min-keyint=1:scenecut=0:bframes=0:no-open-gop=1:log-level=error"}
    return {"crf": str(q["crf"] + 10), "preset": "8", "g": str(gop)}   # libsvtav1


def _stereo(audio, n_samples):
    """AUDIO dict -> float32 [2, n_samples] numpy, trimmed or zero-padded."""
    wav = audio["waveform"]
    if wav.ndim == 3:
        wav = wav[0]
    elif wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.detach().float().cpu()
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav[:2]
    if wav.shape[1] >= n_samples:
        wav = wav[:, :n_samples]
    else:
        wav = F.pad(wav, (0, n_samples - wav.shape[1]))
    return np.ascontiguousarray(wav.numpy())


class _FrameGen:
    """Pairwise frame generation on the GPU via nvvfx (NVIDIA Video Effects).

    Feeds each source frame once and yields the source frames with the
    generated ones between them, all as [H, W, 3] float tensors on the device.
    The last source frame is repeated (multiplier - 1) times so the output is
    exactly `multiplier` x the input length and stays the soundtrack's length.
    """

    def __init__(self, multiplier, width, height, device):
        import nvvfx  # only needed when frame generation is on
        self.m = multiplier
        index = device.index if device.index is not None else torch.cuda.current_device()
        self._device = device
        self._ctx = nvvfx.VideoFrameGeneration(
            mode=nvvfx.VideoFrameGeneration.Mode[FRAME_GEN_MODE],
            automatic_shot_change_detection_enabled=True,
            device=index,
            image_encoding=nvvfx.ImageEncoding.RGB8,
        )
        self._fx = self._ctx.__enter__()
        self._fx.input_width = width
        self._fx.input_height = height
        self._fx.frame_multiplier = multiplier
        self._fx.load()
        self._prev = None
        self._last = None

    def feed(self, frame):
        """One source frame [H, W, 3] on the device -> list of output frames."""
        chw = frame.float().movedim(-1, 0).contiguous()
        out = []
        if self._prev is not None:
            for k in range(1, self.m):
                res = self._fx.run(self._prev, chw, frame_index=k, shot_change=False)
                gen = torch.from_dlpack(res.image).clone()
                if gen.dtype == torch.uint8:
                    gen = gen.float().div_(255.0)
                out.append(gen.movedim(0, -1))
        out.append(frame)
        self._prev = chw
        self._last = frame
        return out

    def finish(self):
        tail = [self._last] * (self.m - 1) if self._last is not None else []
        self.close()
        return tail

    def close(self):
        if self._fx is not None:
            self._fx = None
            self._ctx.__exit__(None, None, None)


class HTCSaveVideo:
    """Encode frames (+ soundtrack) to MP4 in one streaming pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The finished frames, e.g. from H3 Chain Preview."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                  "tooltip": "Frame rate of the frames coming in (H3 renders 24)."}),
                "frame_gen": (list(FRAME_GEN), {"default": "off", "tooltip": (
                    "RTX frame generation inside the save: x2 turns 24 fps into 48. Replaces "
                    "the RTX Video Frame Generation node without ever holding the doubled "
                    "film in RAM. Needs an RTX card and the nvvfx package (installed with "
                    "the NVIDIA RTX nodes).")}),
                "codec": (["auto", "h264", "hevc", "av1"], {"default": "auto", "tooltip": (
                    "auto = H.264, plays everywhere. HEVC and AV1 are smaller at the same "
                    "quality. Encoded on the GPU's NVENC chip when there is one, else on the CPU.")}),
                "quality": (list(QUALITY), {"default": "max", "tooltip": (
                    "Constant quality, not a bitrate. max shows no loss on H3 output; "
                    "archive is near-lossless and large; high is for sharing.")}),
                "filename_prefix": ("STRING", {"default": "video/H3_upsampled",
                    "tooltip": ("Start of the file name in ComfyUI's output folder. A folder "
                                "works too: video/mine.")
                }),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "The soundtrack. Trimmed or padded to the picture."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "filepath", "report")
    FUNCTION = "run"
    CATEGORY = "video"
    OUTPUT_NODE = True

    def run(self, images, fps, frame_gen, codec, quality, filename_prefix,
            audio=None, prompt=None, extra_pnginfo=None):
        if images is None or images.ndim != 4 or images.shape[0] == 0:
            raise ValueError(f"[{TAG}] needs a non-empty IMAGE batch.")
        n_src, h_in, w_in, _ = images.shape
        chunk = max(1, CHUNK_PIXELS // (int(w_in) * int(h_in)))
        parts = (images[s:s + chunk] for s in range(0, int(n_src), chunk))
        return encode_stream(parts, int(n_src), int(w_in), int(h_in), fps, frame_gen, codec,
                             quality, filename_prefix, audio, prompt, extra_pnginfo)


def encode_stream(parts, n_src, w_in, h_in, fps, frame_gen, codec, quality, filename_prefix,
                  audio=None, prompt=None, extra_pnginfo=None):
    """HTCSaveVideo's whole pass over `parts`, an iterable of [k, h_in, w_in, 3+]
    frame batches (any device) totalling `n_src` frames. It is consumed lazily,
    so a producer that decodes as it goes never holds the film. -> the node's
    {"ui", "result"} dict."""
    import comfy.model_management as mm
    import comfy.utils
    import folder_paths

    height, width = h_in - h_in % 2, w_in - w_in % 2

    device = mm.get_torch_device()
    use_gpu = device.type == "cuda"
    mult = FRAME_GEN[frame_gen]
    if mult > 1 and not use_gpu:
        print(f"[{TAG}] frame_gen needs CUDA; saving at the source frame rate.", flush=True)
        mult = 1
    out_fps = float(fps) * mult
    n_out = n_src * mult

    enc_name = _pick_codec(codec, use_gpu, width, height)
    gop = max(1, round(out_fps * KEYFRAME_SECONDS))

    out_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory(), width, height)
    os.makedirs(out_dir, exist_ok=True)
    file = f"{filename}_{counter:05}_.mp4"
    path = os.path.join(out_dir, file)

    t0 = time.perf_counter()
    container = av.open(path, mode="w", options={"movflags": "use_metadata_tags+faststart"})
    fx = None
    writer = None
    ok = False
    try:
        if not _metadata_disabled():
            if prompt is not None:
                container.metadata["prompt"] = json.dumps(prompt)
            for k, v in (extra_pnginfo or {}).items():
                container.metadata[k] = json.dumps(v)

        rate = Fraction(round(out_fps * 1000), 1000)
        vs = container.add_stream(enc_name, rate=rate)
        vs.width, vs.height = width, height
        vs.pix_fmt = "nv12" if enc_name in ("h264_nvenc", "hevc_nvenc", "av1_nvenc", "libx264") else "yuv420p"
        vs.options = _encoder_options(enc_name, quality, gop)
        if enc_name.endswith("_nvenc"):
            vs.bit_rate = 0
        _tag_bt709(vs.codec_context)

        a_packets = []
        if audio is not None and audio.get("waveform") is not None and audio["waveform"].numel():
            sr = int(audio["sample_rate"])
            a_np = _stereo(audio, round(n_src / float(fps) * sr))
            ast = container.add_stream("aac", rate=sr, layout="stereo")
            ast.bit_rate = QUALITY[quality]["audio_kbps"] * 1000
            af = av.AudioFrame.from_ndarray(a_np, format="fltp", layout="stereo")
            af.sample_rate = sr
            af.pts = 0
            a_packets = list(ast.encode(af)) + list(ast.encode(None))

        q = queue.Queue(maxsize=QUEUE_CHUNKS)
        err = []
        writer = threading.Thread(
            target=_mux_loop, args=(q, container, vs, a_packets, float(rate), err),
            name=f"{TAG}-mux", daemon=True)
        writer.start()

        def put(item):
            while True:
                if err:
                    raise err[0]
                try:
                    q.put(item, timeout=0.5)
                    return
                except queue.Full:
                    if not writer.is_alive():
                        raise RuntimeError(f"[{TAG}] muxer stopped") from (err[0] if err else None)

        if mult > 1:
            fx = _FrameGen(mult, width, height, device)
        chunk = max(1, CHUNK_PIXELS // (width * height))
        pbar = comfy.utils.ProgressBar(n_out)
        state = {"prev_thumb": None, "since_key": 0}
        pending = []

        def flush():
            if not pending:
                return
            batch = torch.stack(pending)
            pending.clear()
            nv12, y = rgb_to_nv12(batch)
            keys = _cut_flags(y, state, gop)
            put((nv12.cpu().numpy(), keys))
            pbar.update(len(keys))

        sent = 0
        for part in parts:
            for s in range(0, int(part.shape[0]), chunk):
                mm.throw_exception_if_processing_interrupted()
                src = part[s:s + chunk, :height, :width, :3].to(device, non_blocking=True)
                for i in range(src.shape[0]):
                    pending.extend(fx.feed(src[i]) if fx is not None else [src[i]])
                    if len(pending) >= chunk:
                        flush()
                sent += int(src.shape[0])
        if sent != n_src:
            raise RuntimeError(f"[{TAG}] {sent} frames arrived, {n_src} announced")
        if fx is not None:
            pending.extend(fx.finish())
            fx = None
        flush()
        put(None)
        writer.join()
        if err:
            raise err[0]
        ok = True
    finally:
        if fx is not None:
            fx.close()
        if writer is not None and writer.is_alive():
            while True:   # make room for the stop sentinel, dropping unsent frames
                try:
                    q.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
            writer.join(timeout=30)
        container.close()
        if not ok and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        if use_gpu:
            mm.soft_empty_cache()

    secs = time.perf_counter() - t0
    dur = n_out / out_fps
    report = (f"{file}: {n_out} frames {width}x{height} @ {out_fps:g} fps "
              f"({dur:.1f} s), {enc_name} {quality}"
              f"{', frame gen x' + str(mult) if mult > 1 else ''}; "
              f"saved in {secs:.1f} s ({dur / max(secs, 1e-6):.1f}x real time), "
              f"{os.path.getsize(path) / 1e6:.0f} MB"
              + ("" if a_packets else "; NO AUDIO (the audio input is not wired, or empty)"))
    print(f"[{TAG}] {report}", flush=True)

    from comfy_api.latest import InputImpl
    video = InputImpl.VideoFromFile(path)
    ui = {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}],
          "animated": (True,)}
    return {"ui": ui, "result": (video, path, report)}


def _cut_flags(y, state, gop):
    """Which frames in this batch get a forced keyframe: hard cuts, and the GOP cap."""
    thumbs = F.adaptive_avg_pool2d(y.unsqueeze(1), (16, 16)).flatten(1)
    prev = state["prev_thumb"]
    first = prev is None
    before = torch.cat((thumbs[:1] if first else prev[None], thumbs[:-1]))
    diffs = (thumbs - before).abs().mean(1).tolist()
    state["prev_thumb"] = thumbs[-1]
    keys = []
    for i, d in enumerate(diffs):
        cut = (first and i == 0) or d > CUT_THRESHOLD or state["since_key"] + 1 >= gop
        state["since_key"] = 0 if cut else state["since_key"] + 1
        keys.append(cut)
    return keys


def _mux_loop(q, container, vs, a_packets, fps, err):
    """Encode queued NV12 chunks and mux them, interleaving the audio by time."""
    try:
        n = 0
        ai = 0
        while True:
            item = q.get()
            if item is None:
                break
            nv12, keys = item
            for arr, key in zip(nv12, keys):
                frame = av.VideoFrame.from_ndarray(arr, format="nv12")
                frame.pts = n
                if key:
                    frame.pict_type = av.video.frame.PictureType.I
                for pkt in vs.encode(frame):
                    container.mux(pkt)
                n += 1
                t = n / fps
                while ai < len(a_packets) and float(a_packets[ai].pts * a_packets[ai].time_base) <= t:
                    container.mux(a_packets[ai])
                    ai += 1
        for pkt in vs.encode(None):
            container.mux(pkt)
        for pkt in a_packets[ai:]:
            container.mux(pkt)
    except BaseException as e:  # noqa: BLE001 -- handed back to the caller
        err.append(e)
        while True:   # unblock a producer waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                break


def _tag_bt709(ctx):
    """Tag BT.709 limited range -- what rgb_to_nv12 wrote -- so no player guesses BT.601.

    The enums are newer PyAV; an older one saves the same pixels untagged.
    """
    try:
        from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc, Colorspace
        ctx.color_primaries = ColorPrimaries.BT709
        ctx.color_trc = ColorTrc.BT709
        ctx.colorspace = Colorspace.ITU709
        ctx.color_range = ColorRange.MPEG
    except (ImportError, AttributeError) as e:
        print(f"[{TAG}] colour tags skipped (PyAV too old: {e}); players may guess BT.601",
              flush=True)


def _metadata_disabled():
    try:
        from comfy.cli_args import args
        return bool(args.disable_metadata)
    except Exception:  # noqa: BLE001
        return False


NODE_CLASS_MAPPINGS = {"H3SaveVideo": HTCSaveVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SaveVideo": "H3 Save Video (long-form)"}
