"""MiniMax H3 Combined Reference to Video.

Self-contained: no dependency on any other custom node pack. Everything
here is either (a) a normal runtime import of ComfyUI's own bundled
`comfy_extras.nodes_minimax_h3` module - the same module the native
"MiniMax H3 Image to Video" / "MiniMax H3 Reference to Video" nodes use -
or (b) two small functions vendored, with permission and attribution, from
Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder (MIT License,
Copyright (c) 2026 Adudeguyman):
  - `check_bundle` (refmod_core.py) - validates an H3_REF_MODS bundle.
  - `media_refs` (refmod_nodes.py) - turns an H3_REFS Media Loader bundle
    into presentation items + reference blocks.
  - `KIND_LABEL` (refmods.py).
Both are reproduced near-verbatim below with only the change needed to
drop their reliance on being imported as part of that pack (they now call
this file's own `_core()` helper instead of a local relative import).

WHAT THIS NODE COMBINES
------------------------
One prompt encode that can carry, all at once:
  * first_frame / last_frame keyframes
  * native reference_images (individual, auto-expanding slots - same UI
    as the native H3 node's own `ref_images` Autogrow input)
  * one reference video, one standalone reference audio
  * a Media Loader bundle (`references`, H3_REFS)
  * a RefMod bundle (`mods`, H3_REF_MODS - shared with
    ComfyUI-MiniMaxH3Mod / the Fantastic pack's own loaders)

Earlier versions of this node (and UC's own node, which this used to call
into) treated keyframes and native reference_images as mutually exclusive.
That was never a model-level requirement - `comfy/model_base.py`'s
`MiniMaxH3.extra_conds` sets `minimax_keyframes` and `minimax_refs`
completely independently and feeds both into the same `PackedLayout` - it
was just how the two official nodes happen to be split. This node allows
both together.

WHAT THIS NODE DOES **NOT** COVER
----------------------------------
Multiple reference videos/audios (core's native node supports up to 3 of
each via its own Autogrow inputs; this node takes one of each - add more
via the Media Loader bundle instead), and any of UC-Nodes' own
visual/token/temporal fusion or clip-continuation features, which have no
counterpart in ComfyUI core and were never re-implemented here.
"""

import math

import torch
import torch.nn.functional as F
import node_helpers
import nodes
from comfy_api.latest import ComfyExtension, io

CATEGORY = "PlagueKind/minimax"


# --------------------------------------------------------------------------
# Vendored from silveroxides/ComfyUI-UtilsCollection (AGPLv3), with
# permission from the author. Independent Qwen3-VL presentation resolution:
# lets a reference be shown to the text/vision encoder at a different (and
# usually much smaller) size than the one actually VAE-encoded as a DiT
# reference, so Qwen's vision-token cost doesn't scale with reference
# fidelity. (helpers/encoder_helpers.py + helpers/helper_functions.py,
# verbatim except dropping the "lanczos"/other-method branches this node
# never exercises - only "bicubic" is ever requested here.)
# --------------------------------------------------------------------------

VLM_RESOLUTION_MIN = 256
VLM_RESOLUTION_MAX = 4096
VLM_RESOLUTION_STEP = 32


def resolve_vlm_resolution(value):
    """Return a valid equivalent-square side length, or None for Original."""
    if isinstance(value, bool):
        return None
    try:
        resolution = int(value)
    except (TypeError, ValueError):
        return None
    if resolution < VLM_RESOLUTION_MIN or resolution > VLM_RESOLUTION_MAX:
        return None
    return round(resolution / VLM_RESOLUTION_STEP) * VLM_RESOLUTION_STEP


def vlm_target_dimensions(height, width, resolution):
    """Fit an aspect-preserving target area and align both axes to Qwen's 32px grid."""
    if height < 1 or width < 1:
        raise ValueError("VLM image dimensions must be positive.")
    scale = math.sqrt((resolution * resolution) / (height * width))
    target_height = max(VLM_RESOLUTION_STEP, round(height * scale / VLM_RESOLUTION_STEP) * VLM_RESOLUTION_STEP)
    target_width = max(VLM_RESOLUTION_STEP, round(width * scale / VLM_RESOLUTION_STEP) * VLM_RESOLUTION_STEP)
    return target_height, target_width


def _resize_nchw_bicubic(samples, width, height):
    """(helper_functions.py's resize_nchw, "bicubic" branch only, crop="disabled")"""
    width, height = int(width), int(height)
    source_height, source_width = samples.shape[-2:]
    if (source_width, source_height) == (width, height):
        return samples
    options = {"align_corners": False, "antialias": width < source_width or height < source_height}
    return F.interpolate(samples, size=(height, width), mode="bicubic", **options)


def prepare_vlm_image(image, resolution):
    """Resize a BHWC VLM image to a numeric target, or preserve it for Original."""
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("VLM image must have shape [batch, height, width, channels].")
    target = resolve_vlm_resolution(resolution)
    if target is None:
        return image
    height, width = image.shape[1:3]
    target_height, target_width = vlm_target_dimensions(height, width, target)
    samples = image.movedim(-1, 1)
    return _resize_nchw_bicubic(samples, target_width, target_height).movedim(1, -1)


def prepare_minimax_h3_vlm_video_frames(frames, resolution):
    """Apply the Qwen3-VL resolution to a chronological BHWC frame batch."""
    prepared = [prepare_vlm_image(frames[index:index + 1], resolution) for index in range(frames.shape[0])]
    return torch.cat(prepared, dim=0)


def _core():
    """Import ComfyUI's own MiniMax H3 node module lazily (not at file
    import time) so this file loads even on a ComfyUI build old enough to
    lack MiniMax H3 support - only running the node then fails, clearly."""
    try:
        import comfy_extras.nodes_minimax_h3 as module
        return module
    except Exception as exc:
        raise RuntimeError(
            "This ComfyUI has no native MiniMax H3 support "
            "(comfy_extras.nodes_minimax_h3); update ComfyUI."
        ) from exc


# --------------------------------------------------------------------------
# Vendored from Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder
# (MIT License, Copyright (c) 2026 Adudeguyman), with permission.
# --------------------------------------------------------------------------

KIND_LABEL = {"image": "Picture", "video": "Video", "audio": "Audio"}  # refmods.py


def check_bundle(mods, where: str):
    """A bundle is a list of (mod, strength). Both packs share the link
    type, so say plainly when something else arrives instead of failing
    deep inside with an attribute error. (refmod_core.py, verbatim)"""
    if mods is None:
        return []
    if not isinstance(mods, (list, tuple)):
        raise ValueError(f"{where}: 'mods' is not a RefMod bundle.")
    out = []
    for i, entry in enumerate(mods):
        try:
            mod, strength = entry
        except (TypeError, ValueError):
            raise ValueError(f"{where}: bundle entry {i + 1} is not a (mod, strength) pair.")
        for attr in ("kind", "name", "token_count", "ref_block"):
            if not hasattr(mod, attr):
                raise ValueError(
                    f"{where}: bundle entry {i + 1} ({type(mod).__name__}) has no "
                    f"'{attr}'. It came from a pack this node doesn't understand.")
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            raise ValueError(f"{where}: bundle entry {i + 1} has a non-numeric strength.")
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError(f"{where}: RefMod strength must be between 0 and 1 (entry {i + 1}).")
        out.append((mod, strength))
    return out


def media_refs(references, vae, audio_vae, ref_image_size, width, height, length, counters):
    """Loader media as the encoder's items and the DiT's reference blocks,
    prepared exactly as core's MiniMax H3 Reference to Video prepares its
    own inputs (its helpers do the sizing and the audio encode). The bundle
    is a Media Loader's / Prompt Builder's: pictures, videos, index-paired
    video_audios, standalone audios. Labels continue `counters`, one per
    kind, in that node's order: a video's soundtrack is labelled before it.
    (refmod_nodes.py, adapted to import core's helpers via this file's
    `_core()` instead of a local relative import.)"""
    core = _core()
    _resize, _encode_ref_audio = core._resize, core._encode_ref_audio
    adapt_canvas, temporal_shape = core.adapt_canvas, core.temporal_shape
    CANVAS_MULTIPLE, FPS, REF_IMAGE_SHORT_EDGE = core.CANVAS_MULTIPLE, core.FPS, core.REF_IMAGE_SHORT_EDGE

    if not isinstance(references, dict):
        raise ValueError("'references' is not a Media Loader bundle.")
    frame_count, _t, _a = temporal_shape(length)
    items, blocks, mapping = [], [], []
    seq = lambda key: [v for v in (references.get(key) or []) if v is not None] if key != "video_audios" else list(references.get(key) or [])

    for n, img in enumerate(seq("pictures"), 1):
        h, w = img.shape[1], img.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, "disabled")
        counters["image"] += 1
        mapping.append(f"<Picture {counters['image']}> = picture {n} (media)")
        items.append({"type": "image", "data": resized})
        if vae is not None:
            z = vae.encode(resized)
            blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

    tracks = seq("video_audios")
    videos = references.get("videos") or []
    for n, frames in enumerate(videos, 1):
        if frames is None:
            continue
        soundtrack = tracks[n - 1] if n - 1 < len(tracks) else None
        vh, vw = frames.shape[1], frames.shape[2]
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        k = frames.shape[0]
        if k < 5:
            raise ValueError(f"Reference video {n} needs at least 5 frames (~0.2 s at 24 fps).")
        while k % 17 != 5:
            k -= 1
        frames = frames[:k]
        if soundtrack is not None:
            counters["audio"] += 1
            mapping.append(f"<Audio {counters['audio']}> = soundtrack of video {n} (media)")
            items.append({"type": "audio"})
        counters["video"] += 1
        mapping.append(f"<Video {counters['video']}> = video {n} (media)")
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        items.append({"type": "video", "data": frames[sample_idx],
                      "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        if vae is None:
            continue
        z = vae.encode(frames)
        audio_latent, ref_audio_t = None, 0
        if soundtrack is not None and audio_vae is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
        blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                       "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                       "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

    for n, audio in enumerate(seq("audios"), 1):
        counters["audio"] += 1
        mapping.append(f"<Audio {counters['audio']}> = audio {n} (media)")
        items.append({"type": "audio"})
        if audio_vae is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})
    return items, blocks, mapping


# --------------------------------------------------------------------------
# The node itself.
# --------------------------------------------------------------------------

class PK_MiniMaxH3Combined(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PK_MiniMaxH3Combined",
            display_name="PK MiniMax H3 Combined Reference to Video",
            category=CATEGORY,
            description=(
                "Keyframes, native reference images/video/audio, a Media Loader "
                "bundle, and a RefMod bundle, encoded together in one prompt. "
                "Built directly on ComfyUI's own MiniMax H3 helpers - no other "
                "custom node pack required."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                            "(124 = ~5s; trained range is ~124-362, longer is untested)."),
                io.Vae.Input("vae", optional=True,
                    tooltip="H3 video VAE. Required if first_frame/last_frame are used; "
                            "without it, reference images/videos only condition the text encoder."),
                io.Vae.Input("audio_vae", optional=True,
                    tooltip="H3 audio VAE. Without it, reference audio only conditions the text encoder."),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input("reference_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image",
                            tooltip="Reference image (downscaled to 2048 short edge if larger, never upscaled)"),
                        prefix="reference_image_", min=0, max=9)),
                io.Image.Input("video", optional=True, tooltip="A single reference video (frame batch, 24 fps, 2-15s)."),
                io.Audio.Input("audio", optional=True, tooltip="A single standalone reference audio clip."),
                io.Combo.Input("ref_image_size", options=["match", "max", "none"], default="match",
                    tooltip="'match' scales each reference (down only, keeping aspect) to the generation's "
                            "pixel area. 'max' uses the reference pipeline's 2048px short edge for best "
                            "identity fidelity - reference tokens ride through every sampling step, so 'max' "
                            "can be several times slower. 'none' shows first_frame/last_frame/reference_images/"
                            "video/media pictures to the text encoder only - no VAE-encoded DiT reference block "
                            "(RefMods are unaffected; they're already pre-encoded)."),
                io.Int.Input("vlm_resolution", default=384, min=0, max=4096, step=32,
                    tooltip="Qwen3-VL presentation resolution for still pictures (keyframes and reference "
                            "images), independent of the resolution actually VAE-encoded as the DiT reference. "
                            "0 (or out of 256-4096) keeps the same resolution as the DiT reference."),
                io.Int.Input("vlm_video_resolution", default=384, min=0, max=4096, step=32,
                    tooltip="Qwen3-VL presentation resolution for the reference `video` input's sampled frames. "
                            "Higher values use more visual tokens. 0 (or out of 256-4096) keeps original size."),
                io.Custom("H3_REFS").Input("references", optional=True,
                    tooltip="Media Loader bundle (pictures/videos/video_audios/audios). "
                            "Labelled after first_frame/last_frame/reference_images/video/audio, before RefMods."),
                io.Custom("H3_REF_MODS").Input("mods", optional=True,
                    tooltip="RefMod bundle from a RefMod Stack, or a Prompt Builder's 'mods' output. Labelled last."),
                io.Float.Input("reference_fps", default=24.0, min=1.0, max=120.0,
                    tooltip="Playback rate assumed when reconstructing a RefMod video reference's timestamps."),
                io.Int.Input("max_total_tokens", default=0, min=0, max=2147483647,
                    tooltip="Refuse RefMod bundles over this many reference tokens. 0 = no limit."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="reference_map"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, width=1344, height=768, length=124,
                vae=None, audio_vae=None, first_frame=None, last_frame=None,
                reference_images=None, video=None, audio=None,
                ref_image_size="match", vlm_resolution=384, vlm_video_resolution=384,
                references=None, mods=None,
                reference_fps=24.0, max_total_tokens=0) -> io.NodeOutput:
        core = _core()

        if not math.isfinite(reference_fps) or not 1 <= reference_fps <= 120:
            raise ValueError("reference_fps must be between 1 and 120.")
        if (first_frame is not None or last_frame is not None) and vae is None:
            raise ValueError("Keyframes (first_frame/last_frame) need the H3 video VAE.")

        latent, frame_count = core._empty_av_latent(width, height, length)

        # ref_image_size drives sizing math the same way "match" does; "none"
        # only additionally suppresses the VAE-encoded DiT reference block
        # (media_refs has no "none" of its own, so it gets "match" sizing
        # plus vae=None instead).
        size_mode = "max" if ref_image_size == "max" else "match"
        encode_refs = ref_image_size != "none"

        keyframes = []          # -> metadata["minimax_keyframes"]; each holds a pixel "image" until encoded below
        keyframe_vlm_images = []  # Qwen-facing copies (vlm_resolution), only used when nothing else needs minimax_ref_items
        ref_items = []          # tokenizer presentation, in order (native refs/video/audio/media/RefMods)
        ref_blocks = []          # -> metadata["minimax_refs"], same order
        mapping = []
        counters = {"image": 0, "video": 0, "audio": 0}

        # ---- keyframes ----
        if first_frame is not None:
            img = core._resize(first_frame[:1], width, height, "disabled")
            keyframe_vlm_images.append(prepare_vlm_image(img, vlm_resolution))
            counters["image"] += 1
            mapping.append(f"<Picture {counters['image']}> = first frame")
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            img = core._resize(last_frame[:1], width, height, "center")
            keyframe_vlm_images.append(prepare_vlm_image(img, vlm_resolution))
            counters["image"] += 1
            mapping.append(f"<Picture {counters['image']}> = last frame")
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        # ---- native reference images (Autogrow: reference_image_0, reference_image_1, ...) ----
        for img in (reference_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if size_mode == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, core.REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(core.CANVAS_MULTIPLE, round(w * scale / core.CANVAS_MULTIPLE) * core.CANVAS_MULTIPLE)
            th = max(core.CANVAS_MULTIPLE, round(h * scale / core.CANVAS_MULTIPLE) * core.CANVAS_MULTIPLE)
            resized = core._resize(img[:1], tw, th, "disabled")
            counters["image"] += 1
            mapping.append(f"<Picture {counters['image']}> = reference image")
            ref_items.append({"type": "image", "data": prepare_vlm_image(resized, vlm_resolution)})
            if vae is not None and encode_refs:
                z = vae.encode(resized)
                ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        # ---- reference video (single) ----
        if video is not None:
            vh, vw = video.shape[1], video.shape[2]
            cw, ch = core.adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(core.CANVAS_MULTIPLE, round(vw / core.CANVAS_MULTIPLE) * core.CANVAS_MULTIPLE)
                ch = max(core.CANVAS_MULTIPLE, round(vh / core.CANVAS_MULTIPLE) * core.CANVAS_MULTIPLE)
            frames = core._resize(video, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("Reference video needs at least 5 frames (~0.2s at 24 fps).")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            counters["video"] += 1
            mapping.append(f"<Video {counters['video']}> = video")
            sample_idx = list(range(0, frames.shape[0], core.FPS // 2))
            ref_items.append({"type": "video",
                               "data": prepare_minimax_h3_vlm_video_frames(frames[sample_idx], vlm_video_resolution),
                               "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            if vae is not None and encode_refs:
                z = vae.encode(frames)
                ref_blocks.append({"kind": "video", "latent_t": z.shape[2], "latent_h": ch // 16,
                                    "latent_w": cw // 16, "ref_audio_t": 0, "latent": z, "audio_latent": None})

        # ---- standalone reference audio (single) ----
        if audio is not None:
            counters["audio"] += 1
            mapping.append(f"<Audio {counters['audio']}> = audio")
            ref_items.append({"type": "audio"})
            if audio_vae is not None:
                audio_latent, ref_audio_t = core._encode_ref_audio(audio_vae, audio)
                ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        # ---- Media Loader bundle ----
        if references is not None:
            m_items, m_blocks, m_mapping = media_refs(
                references, vae if encode_refs else None, audio_vae, size_mode, width, height, length, counters)
            ref_items.extend(m_items)
            ref_blocks.extend(m_blocks)
            mapping.extend(m_mapping)

        # ---- RefMod bundle ----
        active = [(m, s) for m, s in check_bundle(mods, "Combined MiniMax H3 node") if s > 0]
        if max_total_tokens:
            total = sum(m.token_count for m, _s in active)
            if total > max_total_tokens:
                raise ValueError(
                    f"RefMods require {total} tokens; the limit is {max_total_tokens}. "
                    "Lower a weight, drop a pick, or raise the limit."
                )
        decoded = {}
        for mod, strength in active:
            block = mod.ref_block(strength)
            if block is None:
                continue
            block["refmod"] = True
            kind = block["kind"]
            if kind not in counters:
                raise ValueError(f"RefMod '{mod.name}' has kind '{kind}', which this node cannot label.")
            counters[kind] += 1
            mapping.append(f"<{KIND_LABEL[kind]} {counters[kind]}> = {mod.name}")
            item = {"type": kind}
            if kind != "audio":
                if vae is None:
                    raise ValueError(f"RefMod '{mod.name}' needs the H3 video VAE to be shown to the encoder.")
                key = (id(mod), round(float(strength), 4))
                if key not in decoded:
                    pixels = vae.decode(block["latent"])
                    if pixels.ndim == 5 and pixels.shape[0] == 1:
                        pixels = pixels[0]
                    decoded[key] = pixels.cpu()
                pixels = decoded[key]
                if kind == "image":
                    item["data"] = pixels[:1].clone()
                else:
                    times = [i / 2 for i in range(math.ceil(pixels.shape[0] * 2 / reference_fps))]
                    idx = [min(round(t * reference_fps), pixels.shape[0] - 1) for t in times]
                    item["data"] = pixels[idx].clone()
                    item["timestamps"] = times
            ref_items.append(item)
            ref_blocks.append(block)
        del decoded

        # ---- tokenize + encode ----
        # The tokenizer's `images=` path is only consulted when
        # `minimax_ref_items` is empty/absent, so once anything besides a
        # plain keyframe is present, keyframe pixels have to ride in
        # minimax_ref_items too (as ordinary "image" items) to still reach
        # the text/vision encoder.
        if ref_items:
            keyframe_items = [{"type": "image", "data": img} for img in keyframe_vlm_images]
            tokens = clip.tokenize(prompt, minimax_ref_items=keyframe_items + ref_items)
        else:
            tokens = clip.tokenize(prompt, images=keyframe_vlm_images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        metadata_update = {}
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            metadata_update["minimax_keyframes"] = keyframes
        if ref_blocks:
            metadata_update["minimax_refs"] = ref_blocks
        if metadata_update:
            conditioning = node_helpers.conditioning_set_values(conditioning, metadata_update)

        return io.NodeOutput(conditioning, latent, "\n".join(mapping) or "No references.")


class PK_MiniMaxH3CombinedExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PK_MiniMaxH3Combined]


async def comfy_entrypoint() -> PK_MiniMaxH3CombinedExtension:
    return PK_MiniMaxH3CombinedExtension()
