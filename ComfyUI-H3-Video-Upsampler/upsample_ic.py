# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""In-context (IC) upsample: the second re-draw method beside Regen2K.

After NVlabs' Open-H3-Regenerate-2K-IC, as our own code on core pieces (nothing
imported from their repo). Regen2K starts from the 3D-upscaled draft and never
sees the draft again; IC keeps each chunk's low-res draft IN CONTEXT as
<Video 1> of the Ref2VA transformer, so faces, lettering and mouths stay the
draft's own (side by side on a dialogue clip, 2026-11-09: same people, same
signature; Regen2K redrew both).

What it changes in tiled_upsampler.upsample (everything else -- chunk plan,
stepped joins, 3D upscale, memory budgets, low-VRAM patches, sparse attention,
the file node's streaming -- is the Regen2K path unchanged):

  * the model: HyperFlow 8-step (merged weights + its two-time embedder),
    LMS Sharp at 1.0, then the IC LoRA -- `patch_model`; Kijai's rank-21
    resize (200 MB, ComfyUI layout) is preferred: measured within run-to-run
    noise of the full r384 file (2026-11-09, a 12 s dialogue clip);
  * the sigmas: HyperFlow's grid shifted by 12, from index 5 (up3:
    0.835 / 0.697 / 0.469 / 0), CFG 1, euler;
  * the conditioning, per chunk, built before the DiT is staged
    (`ChunkConds.build`): the fixed task sentence, Qwen3-VL shown the chunk's
    draft at 2 fps, and the chunk's draft latent in minimax_refs (the low-res
    latent the upsampler already holds -- no extra encode unless the draft is
    bigger than H3's reference canvas);
  * the audio rows: the clip's own soundtrack, held clean. IC was trained on
    silent rows, but with them mouths only follow the soft draft; with the
    soundtrack they shape to the speech (2026-11-09, same cost). Encoded
    silence only when there is no soundtrack to hear.

The LoRA files ship in Diffusers names; they are mapped onto core's fused
qkv_proj (row offsets) and fc1 (Diffusers [value; gate] -> core [gate; value]).
tools/check_upsample_ic.py checks the mapping numerically.
"""
from __future__ import annotations

import json
import math
import re

import torch

TAG = "H3VideoUpsampler IC"
IC_LORA = "open_h3_regenerate_2k_ic_lora.safetensors"            # NVlabs, Diffusers names, 8.3 GB
IC_LORA_SMALL = "minimax_h3_open_regenerate_2k_ic_lora_avg_rank_21_bf16.safetensors"  # Kijai, 200 MB
HYPERFLOW_LORA = "minimax_h3_hyperflow_8step_v1.0.safetensors"
SHARP_LORA = "minimax_h3_lms_v1.0_r64.safetensors"
SHARP_STRENGTH = 1.0
SIGMA_START_INDEX = 5          # up3 in HyperFlow's 9-point grid
CHUNK_FRAMES = 119             # 7 x 17: nearest the 124-frame training window on the key grid
QWEN_EVERY = 12                # Qwen sees the draft at 2 fps (core ReferenceToVideo)
TASK = ('Enhance <Video 1> into a high-quality version of the same video. '
        'Preserve the scene, objects, camera motion, timing, and composition of every frame. '
        'Restore fine details and remove degradation. Do not continue or change the scene.')
METHODS = ("regen2k", "ic")
METHOD_TIP = ("regen2k = the Regen2K LoRA re-draws detail from the enlarged clip (faster). "
              "ic = in-context: each piece's original frames stay in view while it is "
              "re-drawn and it hears the soundtrack, so faces, text and mouths stay the "
              "clip's own (slower). ic needs audio_vae for the soundtrack and, in "
              "models/loras, an IC LoRA "
              f"({IC_LORA_SMALL}, 200 MB) + {HYPERFLOW_LORA} (+ {SHARP_LORA}). Under ic, lora "
              "picks the IC file when it names one; redraw_steps 2 starts one step later (less re-drawn, about a third faster); redraw_start is ignored. "
              "Tiles only when a piece is too big for the GPU; each tile then sees its own "
              "window of the original frames.")

_KEY = re.compile(r"^transformer\.(?P<m>.+)\.lora_(?P<x>[AB])\.weight$")


def is_ic(method):
    return str(method or "").strip().lower().startswith("ic")


# --- LoRA key mapping (Diffusers -> core) -----------------------------------

def _target(module, shapes):
    """Diffusers module path -> [(core weight key, offset or None, B row slice or None)]."""
    name = (module.replace("transformer_blocks.", "blocks.", 1)
            if module.startswith("transformer_blocks.") else module)
    name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", 1)
    root, _, leaf = name.rpartition(".attn.")
    if leaf in ("to_q", "to_k", "to_v"):
        key = "diffusion_model.%s.attn.qkv_proj.weight" % root
        n = shapes[key][0] // 3
        i = ("to_q", "to_k", "to_v").index(leaf)
        return [(key, (0, i * n, n), None)]
    if leaf == "to_out.0":
        return [("diffusion_model.%s.attn.out_proj.weight" % root, None, None)]
    if name.endswith(".ff.net.0.proj"):
        key = "diffusion_model.%s.mlp.fc1.weight" % name[:-len(".ff.net.0.proj")]
        ffn = shapes[key][0] // 2
        # Diffusers rows [value; gate] -> core rows [gate; value]
        return [(key, (0, ffn, ffn), slice(0, ffn)), (key, (0, 0, ffn), slice(ffn, 2 * ffn))]
    if name.endswith(".ff.net.2"):
        return [("diffusion_model.%s.mlp.fc2.weight" % name[:-len(".ff.net.2")], None, None)]
    raise KeyError("no core target for LoRA module %r" % module)


def diffusers_lora_patches(path, shapes, skip=("time_embedder.", "endpoint_time_embedder."),
                           dtype=torch.bfloat16):
    """A Diffusers-named H3 LoRA (alpha == rank) -> (core patches, metadata)."""
    import comfy.lora
    from safetensors import safe_open
    sd, to_load = {}, {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        mods = {}
        for k in f.keys():
            m = _KEY.match(k)
            if not m:
                raise KeyError("unexpected key %r in %s" % (k, path))
            mods.setdefault(m["m"], {})[m["x"]] = k
        for i, (module, ab) in enumerate(sorted(mods.items())):
            if module.startswith(skip):
                continue
            a = f.get_tensor(ab["A"]).to(dtype)
            b = f.get_tensor(ab["B"]).to(dtype)
            for j, (key, offset, rows) in enumerate(_target(module, shapes)):
                if key not in shapes:
                    raise KeyError("the model has no %s (from %s)" % (key, module))
                x = "ic%d_%d" % (i, j)
                sd[x + ".lora_A.weight"] = a
                sd[x + ".lora_B.weight"] = (b[rows] if rows is not None else b).contiguous()
                to_load[x] = (key, offset) if offset is not None else key
    patches = comfy.lora.load_lora(sd, to_load, log_missing=False)
    if len(patches) != len(to_load):
        raise RuntimeError("%s: %d of %d LoRA targets loaded" % (path, len(patches), len(to_load)))
    return patches, meta


def endpoint_deltas(path):
    """HyperFlow's time-embedder LoRAs: {'time_embedder'|'endpoint_time_embedder':
    {'linear_1': dW, 'linear_2': dW}}."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        scale = float(meta.get("lora_alpha", 1)) / float(meta.get("lora_rank", 1))
        for k in f.keys():
            m = _KEY.match(k)
            if not m or m["x"] != "A":
                continue
            mod = m["m"]
            for top in ("endpoint_time_embedder", "time_embedder"):
                if mod.startswith(top + "."):
                    a = f.get_tensor(k).float()
                    b = f.get_tensor(k[:-len("lora_A.weight")] + "lora_B.weight").float()
                    out.setdefault(top, {})[mod[len(top) + 1:]] = scale * (b @ a)
                    break
    return out, meta


def hyperflow_sigmas(meta):
    raw = torch.tensor(json.loads(meta["hyperflow_sigmas"]), dtype=torch.float64)
    shift = float(meta["hyperflow_video_shift"])
    return (shift * raw / (1.0 + (shift - 1.0) * raw)).tolist()


# --- HyperFlow two-time embedder --------------------------------------------

class TwoTimeEmbedder(torch.nn.Module):
    """Stands in for core's TimeEmbedder: emb(t) + gate * (emb_endpoint(r) - emb(t)),
    r = 1 - next sigma for the rows being generated, r = t for the rest."""

    def __init__(self, base_weights, deltas, gate, freq_dim):
        super().__init__()
        w1, b1, w2, b2 = (t.detach().float().cpu().clone() for t in base_weights)
        own = deltas.get("time_embedder", {})
        ep = deltas.get("endpoint_time_embedder", {})
        z1, z2 = torch.zeros_like(w1), torch.zeros_like(w2)
        self.main = (w1 + own.get("linear_1", z1), b1, w2 + own.get("linear_2", z2), b2)
        self.endp = (w1 + ep.get("linear_1", z1), b1, w2 + ep.get("linear_2", z2), b2)
        self.gate = float(gate)
        self.freq_dim = int(freq_dim)
        self.step = None          # (t of the generated rows, r for them), set by the wrapper
        self._dev = {}
        self.misses = 0

    def _on(self, device):
        if device not in self._dev:
            self._dev = {device: (tuple(t.to(device) for t in self.main),
                                  tuple(t.to(device) for t in self.endp))}
        return self._dev[device]

    def _emb(self, t, p):
        # core TimeEmbedder's projection: fp32, cos before sin, t in [0, 1]
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        e = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        h = torch.nn.functional.silu(torch.nn.functional.linear(e, p[0], p[1]))
        return torch.nn.functional.linear(h, p[2], p[3])

    def forward(self, t):
        main, endp = self._on(t.device)
        t = t.to(torch.float32)
        r = t.clone()
        if self.step is not None:
            t_gen, r_gen = self.step
            hit = (t - t_gen).abs() < 1e-6
            if int(hit.sum()) != 1:
                self.misses += 1
            r[hit] = r_gen
        else:
            self.misses += 1
        t_emb = self._emb(t, main)
        return t_emb + self.gate * (self._emb(r, endp) - t_emb)


def step_wrapper(embedder):
    """DIFFUSION_MODEL wrapper: tells the embedder this call's t and next sigma."""
    def wrapper(executor, *args, **kwargs):
        timestep = args[1]
        opts = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        sv = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        ss = opts.get("sample_sigmas")
        nxt = 0.0
        if ss is not None:
            ss = ss.flatten().float().cpu()
            i = int((ss - float(sv)).abs().argmin())
            nxt = float(ss[i + 1]) if i + 1 < ss.shape[0] else 0.0
        embedder.step = (float(1.0 - sv), 1.0 - nxt)
        try:
            return executor(*args, **kwargs)
        finally:
            embedder.step = None
    return wrapper


def _base_time_weights(model):
    import comfy.utils
    dm = model.model.diffusion_model
    out = []
    for name in ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias"):
        key = "diffusion_model.time_embedder." + name
        bk = getattr(model, "backup", {}).get(key)
        out.append(bk.weight if bk is not None else comfy.utils.get_attr(dm, "time_embedder." + name))
    return out


# --- the model ----------------------------------------------------------------

_CACHE = {}       # mapped patches while one run is set up; release() drops them (~5.5 GB)


def release():
    """Drop the mapped LoRA weights (on 32 GB machines they matter)."""
    _CACHE.clear()


def _file_key(path):
    import os
    st = os.stat(path)
    return (path, st.st_size, int(st.st_mtime))


def _cached(path, shapes, kind):
    key = (kind,) + _file_key(path) + (len(shapes),)
    hit = _CACHE.get(key)
    if hit is None:
        if kind == "patches":
            hit = diffusers_lora_patches(path, shapes)
        else:
            hit = endpoint_deltas(path)
        # keep only the current file set (IC is ~4 GB in bf16)
        for k in [k for k in _CACHE if k[0] == kind]:
            del _CACHE[k]
        _CACHE[key] = hit
    return hit


def is_ic_lora(name):
    """A file name that is an IC LoRA (the full Diffusers one or a ComfyUI resize)."""
    return "regenerate_2k_ic" in str(name or "").lower()


def pick_ic_lora(wanted=None):
    """The IC LoRA file name: `wanted` when it names an IC LoRA in models/loras,
    else the 200 MB resize (Kijai's rank-21), else the full file. None if none."""
    import folder_paths
    have = set(folder_paths.get_filename_list("loras"))
    if wanted and is_ic_lora(wanted) and wanted in have:
        return wanted
    for name in (IC_LORA_SMALL, IC_LORA):
        if name in have:
            return name
    return None


def lora_paths(wanted=None):
    """(ic, hyperflow, sharp or None) full paths; a missing required file raises."""
    import folder_paths
    have = set(folder_paths.get_filename_list("loras"))
    ic_name = pick_ic_lora(wanted)
    missing = ([] if ic_name else [f"{IC_LORA_SMALL} (or {IC_LORA})"]) + (
        [] if HYPERFLOW_LORA in have else [HYPERFLOW_LORA])
    if missing:
        raise RuntimeError(
            f"{TAG}: method ic needs {', '.join(missing)} in models/loras "
            f"(the Open-H3-Regenerate-2K-IC LoRA and the HyperFlow 8-step weights it was "
            f"trained on). Put them there, or pick method regen2k.")
    sharp = folder_paths.get_full_path("loras", SHARP_LORA) if SHARP_LORA in have else None
    return (folder_paths.get_full_path_or_raise("loras", ic_name),
            folder_paths.get_full_path_or_raise("loras", HYPERFLOW_LORA), sharp)


NEEDS_DIT = ("method ic needs the plain official ref2va DiT "
             "(UNETLoader: minimax_h3_ref2va_int8_convrot or another minimax_h3_ref2va_*, no "
             "LoRAs). The fl2va / pruned builds and curve-form merges have no time embedder, "
             "which IC's HyperFlow step needs.")


def model_has_time_embedder(model):
    """True / False for a ModelPatcher's DiT, None when it cannot be told."""
    try:
        dm = (model.get_model_object("diffusion_model") if hasattr(model, "get_model_object")
              else model.model.diffusion_model)
    except Exception:
        return None
    return getattr(dm, "time_embedder", None) is not None


def preflight(clip, lora=None, needs_clip=True, model=None):
    """Fail before any work: the DiT, the clip input and the files method ic needs."""
    if model is not None and model_has_time_embedder(model) is False:
        raise RuntimeError(f"{TAG}: the wired model has no time embedder. {NEEDS_DIT} "
                           "Nothing was run.")
    if needs_clip and clip is None and load_task_cond() is None:
        raise RuntimeError(f"{TAG}: method ic needs the clip input wired (CLIPLoader with the "
                           f"H3 text encoder, type minimax), or {TASK_COND_FILE} beside "
                           "upsample_ic.py. Nothing was run.")
    lora_paths(lora)


def _is_diffusers(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return any(k.startswith("transformer.") for k in f.keys())


def comfy_lora_patches(model, path):
    """An IC LoRA already in ComfyUI's H3 layout (e.g. Kijai's resize) -> (patches, meta)."""
    import comfy.lora
    import comfy.utils
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    patches = comfy.lora.load_lora(sd, key_map, log_missing=False)
    want = len({k[:-len(".lora_A.weight")] for k in sd if k.endswith(".lora_A.weight")})
    if len(patches) != want:
        raise RuntimeError(f"{TAG}: {path}: {len(patches)} of {want} LoRA targets loaded")
    return patches, meta


def patch_model(model, ic_lora=None):
    """HyperFlow (+ two-time embedder), LMS Sharp, IC LoRA on a clone of `model`.
    `ic_lora` picks the IC file (pick_ic_lora). -> (model, sigmas list, note)."""
    import os

    import comfy.lora
    import comfy.patcher_extension
    import comfy.utils

    ic_path, hf_path, sharp_path = lora_paths(ic_lora)
    m = model.clone()
    shapes = {k: tuple(v.shape) for k, v in m.model.state_dict().items()}
    if "diffusion_model.time_embedder.proj_in.weight" not in shapes:
        raise RuntimeError(f"{TAG}: this DiT has no time_embedder. {NEEDS_DIT}")
    hf_patches, hf_meta = _cached(hf_path, shapes, "patches")
    if hf_meta.get("hyperflow") != "true":
        raise RuntimeError(f"{TAG}: {HYPERFLOW_LORA} is not a HyperFlow weights file")
    deltas, _ = _cached(hf_path, shapes, "deltas")
    n_hf = len(m.add_patches(hf_patches, 1.0))
    te = m.model.diffusion_model.time_embedder
    emb = TwoTimeEmbedder(_base_time_weights(m), deltas, hf_meta.get("hyperflow_gate", 1.0),
                          te.freq_dim)
    m.add_object_patch("diffusion_model.time_embedder", emb)
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                           "htc_upsample_ic_hyperflow", step_wrapper(emb))
    n_sh = 0
    if sharp_path is not None and SHARP_STRENGTH > 0:
        sd = comfy.utils.load_torch_file(sharp_path, safe_load=True)
        # the published r64's 8-D AdaLN curve branches are all zero: dropped
        curve = {k[:-len("lora_A.weight")] for k, v in sd.items()
                 if k.endswith("lora_A.weight") and v.shape[-1] == 8}
        sd = {k: v for k, v in sd.items() if not any(k.startswith(c) for c in curve)}
        key_map = comfy.lora.model_lora_keys_unet(m.model, {})
        sh = comfy.lora.load_lora(sd, key_map, log_missing=False)
        sh = {k: v for k, v in sh.items()
              if "time_embedder" not in (k[0] if isinstance(k, tuple) else k)}
        n_sh = len(m.add_patches(sh, float(SHARP_STRENGTH)))
    if _is_diffusers(ic_path):
        ic_patches, ic_meta = _cached(ic_path, shapes, "patches")
    else:
        ic_patches, ic_meta = comfy_lora_patches(m, ic_path)
    n_ic = len(m.add_patches(ic_patches, 1.0))
    grid = hyperflow_sigmas(hf_meta)
    sigmas = [float(s) for s in grid[SIGMA_START_INDEX:]]
    note = (f"ic (HyperFlow {n_hf} + endpoint gate {emb.gate:g}, "
            + (f"Sharp {n_sh} @ {SHARP_STRENGTH:g}, " if n_sh else "no Sharp, ")
            + f"IC {n_ic} from {os.path.basename(ic_path)}, "
            + f"step {ic_meta.get('training_step', '?')})")
    m._htc_ic_embedder = emb
    return m, sigmas, note


def ic_sigmas(sigmas, steps=None):
    """The last `steps` steps of the IC schedule (1-3; None or more = all 3).
    2 = 0.697 / 0.469 / 0: starts one grid point later, so less is re-drawn
    and it runs about a third faster. HyperFlow's step embedder is told the
    next sigma on every call, so any tail of its grid is a valid schedule."""
    sig = [float(s) for s in sigmas]
    full = len(sig) - 1
    n = full if steps is None else max(1, min(full, int(steps)))
    return sig[full - n:]


def embedder_misses(model):
    emb = getattr(model, "_htc_ic_embedder", None)
    return int(emb.misses) if emb is not None else 0


# --- per-chunk conditioning --------------------------------------------------

TASK_COND_FILE = "upsample_ic_task.safetensors"   # beside this module


def task_cond_path():
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), TASK_COND_FILE)


def save_task_cond(cond, path=None):
    """The text-only encoding of the task prompt -> a small file, so ic runs
    without the text encoder (and its ~26 GB in RAM)."""
    from safetensors.torch import save_file
    (emb, d), = cond
    tensors, meta = {"emb": emb.detach().float().cpu().contiguous()}, {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            tensors["d." + k] = v.detach().cpu().contiguous()
        else:
            meta[k] = json.dumps(v)
    meta["prompt"] = TASK + "\n\nScene description: "
    save_file(tensors, path or task_cond_path(), metadata=meta)


def load_task_cond(path=None):
    """-> the cached task conditioning, or None when the file is missing."""
    import os
    path = path or task_cond_path()
    if not os.path.exists(path):
        return None
    from safetensors import safe_open
    d = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        emb = f.get_tensor("emb")
        for k in f.keys():
            if k.startswith("d."):
                d[k[2:]] = f.get_tensor(k)
    meta.pop("prompt", None)
    for k, v in meta.items():
        d[k] = json.loads(v)
    return [[emb, d]]


class ChunkConds:
    """Builds each chunk's IC conditioning from its low-res draft latent, before
    the DiT is staged. The draft latent always rides in minimax_refs as
    <Video 1>. With `clip` wired, Qwen3-VL is also shown the chunk's draft at
    2 fps (NVlabs' recipe); without it, the task prompt's cached text-only
    encoding (TASK_COND_FILE) is used and no text encoder loads at all."""

    def __init__(self, clip, decode, encode=None, caption=""):
        self.cached = None
        if clip is None:
            self.cached = load_task_cond()
            if self.cached is None:
                raise RuntimeError(f"{TAG}: method ic needs the clip input wired "
                                   "(CLIPLoader with the H3 text encoder, type minimax), or "
                                   f"{TASK_COND_FILE} beside upsample_ic.py.")
        self.clip = clip
        self.decode = decode
        self.encode = encode
        self.prompt = TASK + "\n\nScene description: " + (caption or "")
        self.resized = 0
        self.note = ("the text encoder sees each piece" if clip is not None
                     else "cached task text, no text encoder")

    def build(self, raw):
        """raw [1, 24, T, h, w] draft latent (starting on a key step) -> conditioning."""
        import node_helpers
        from comfy_extras.nodes_minimax_h3 import _resize, adapt_canvas
        from . import tiled_upsampler as _tu

        t = int(raw.shape[2])
        n = _tu.frames_in(t)
        hh, ww = int(raw.shape[3]) * _tu.SPATIAL, int(raw.shape[4]) * _tu.SPATIAL
        cw, ch = adapt_canvas(ww, hh)
        lat = raw
        big = ww * hh > cw * ch
        px = None
        if self.clip is not None or big:
            px = self.decode(raw)
            px = px.reshape(-1, px.shape[-3], px.shape[-2], px.shape[-1])[:n, ..., :3]
        if big:
            # bigger than H3's reference canvas: as core shows a reference video,
            # shrunk to the canvas (and encoded there)
            px = _resize(px, cw, ch, "disabled")
            if self.encode is None:
                raise RuntimeError(f"{TAG}: the draft is above the reference canvas and "
                                   "no encoder was given")
            lat = _tu.encode_steps(self.encode, px, t)
            self.resized += 1
        if self.clip is not None:
            idx = list(range(0, n, QWEN_EVERY))
            items = [{"type": "video", "data": px[idx].float().cpu(),
                      "timestamps": [i / 2.0 for i in range(len(idx))]}]
            tokens = self.clip.tokenize(self.prompt, minimax_ref_items=items)
            cond = self.clip.encode_from_tokens_scheduled(tokens)
        else:
            cond = [[e, dict(d)] for e, d in self.cached]
        lat = lat.detach().cpu()
        blk = {"kind": "video", "latent_t": int(lat.shape[2]), "latent_h": int(lat.shape[3]),
               "latent_w": int(lat.shape[4]), "ref_audio_t": 0, "latent": lat,
               "audio_latent": None, _tu.TILE_CROP_KEY: True}
        del px
        return node_helpers.conditioning_set_values(cond, {"minimax_refs": [blk]})


def silent_audio_latent(audio_vae, frames, audio_t, dtype, device):
    """Encoded silence for `frames` video frames, fitted to `audio_t` steps (as
    trained); a zero latent when no audio VAE is wired."""
    if audio_vae is None:
        return torch.zeros((1, 32, 2, int(audio_t)), dtype=dtype, device=device), "zero latent"
    sr = int(getattr(audio_vae, "audio_sample_rate", 44100))
    n = round((int(frames) + 1) / 24.0 * sr)
    wav = torch.zeros((1, 2, n + (-n % 800)))
    z = audio_vae.encode(wav.movedim(1, -1))
    have = int(z.shape[-1])
    if have >= int(audio_t):
        z = z[..., :int(audio_t)]
    else:
        z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], int(audio_t) - have)], dim=-1)
    return z.to(device=device, dtype=dtype), "encoded silence"
