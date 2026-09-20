"""MiniMax H3 Combined Image/Reference to Video.

One node that folds together the reference paths from two separate,
independently-installed packs, instead of you wiring both native H3 nodes
and picking one:

  * silveroxides/ComfyUI-UtilsCollection
      -> UC_AdvancedMiniMaxH3ImageToVideo
         (nodes/encoder_nodes.py, helpers/encoder_helpers.py)
      Gives us: keyframes (first_frame / last_frame), native H3
      reference_images, a reference `video`, and reference `audio`.

  * Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder
      -> MiniMaxH3FantasticRefModTextEncode
         (refmod_nodes.py, refmod_core.py, refmods.py)
      Gives us: the Media Loader `references` bundle (H3_REFS) and the
      `mods` RefMod bundle (H3_REF_MODS, shared with Luisa's
      ComfyUI-MiniMaxH3Mod).

Both of those already agree on the same underlying contracts ComfyUI core
uses (`clip.tokenize(prompt, minimax_ref_items=[...])` and the
`minimax_keyframes` / `minimax_refs` conditioning metadata), so this node
does not reimplement either of them: it locates the already-loaded modules
at runtime and calls their real functions directly. That also means the
result is only as correct as those two packs' own code.

WHAT THIS NODE DOES **NOT** COVER, ON PURPOSE
----------------------------------------------
`UC_AdvancedMiniMaxH3ImageToVideo` also supports spatial/token visual
fusion (`visual_fusion_config`), Picture timestamp formatting and
Video-motion guidance (`media_config`), clip-continuation, and temporal
fusion. That machinery is several hundred lines of bespoke consensus/fusion
math tightly closed over private state in UC's helper, and it cannot be
safely spliced into a second reference pipeline without the ability to test
against a live MiniMax H3 model. If you need those specific features, use
UC's own `UC_AdvancedMiniMaxH3ImageToVideo` node directly instead of this
one. Everything else from both packs — keyframes, native reference images,
a reference video, reference audio, the Media Loader bundle, and RefMods —
is genuinely combined here into a single prompt encode.

Native H3 rule preserved: keyframes (first_frame/last_frame) and native
`reference_images` are still mutually exclusive with each other, same as in
UC's own node.

V3 SCHEMA NOTE
--------------
`reference_images` uses Autogrow (individual, auto-expanding/collapsing
IMAGE slots — same UX as the native H3 node), and Autogrow has no V1
equivalent, so this node is V3 schema (`io.ComfyNode`) instead of the old
INPUT_TYPES/FUNCTION style. It registers through `comfy_entrypoint()` /
`ComfyExtension`, not `NODE_CLASS_MAPPINGS`. If your pack's `__init__.py`
aggregates nodes by importing each module's `NODE_CLASS_MAPPINGS`, point it
at this module's `comfy_entrypoint()` instead (or register this file's
extension directly with ComfyUI's loader) or this node will silently not
appear.
"""

import math
import sys

from comfy_api.latest import ComfyExtension, io

CATEGORY = "PlagueKind/minimax"


# --------------------------------------------------------------------------
# Locate the two host packs at runtime, regardless of what folder name they
# were cloned/installed under. Both packs are already fully imported by
# ComfyUI at startup (that's how their own nodes work), so by the time this
# node's `execute()` runs, their modules exist somewhere in sys.modules.
# We find them by a distinctive attribute rather than by guessing the
# top-level package name.
# --------------------------------------------------------------------------

def _find_module(distinctive_attr):
    for mod in list(sys.modules.values()):
        # Check the module's real __dict__ instead of hasattr(): some
        # modules (notably torch.ops) implement __getattr__ that lazily
        # fabricates an object for *any* attribute name without raising,
        # which makes hasattr() return a false positive and hands back
        # the wrong module entirely.
        if mod is not None and distinctive_attr in vars(mod):
            return mod
    return None


class _Deps:
    """Lazily resolved, cached references into the two host packs."""

    def __init__(self):
        self.uc = None          # ComfyUI-UtilsCollection: helpers/encoder_helpers.py
        self.fr_nodes = None     # Fantastic pack: refmod_nodes.py
        self.fr_core = None      # Fantastic pack: refmod_core.py
        self.fr_mods = None      # Fantastic pack: refmods.py

    def resolve(self):
        if self.uc is None:
            self.uc = _find_module("minimax_h3_empty_av_latent")
            if self.uc is None:
                raise RuntimeError(
                    "Combined MiniMax H3 node: could not find "
                    "silveroxides/ComfyUI-UtilsCollection (looked for "
                    "'minimax_h3_empty_av_latent'). Make sure that custom "
                    "node pack is installed and enabled, then restart ComfyUI."
                )
        if self.fr_nodes is None:
            self.fr_nodes = _find_module("media_refs")
            if self.fr_nodes is None:
                raise RuntimeError(
                    "Combined MiniMax H3 node: could not find "
                    "Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder "
                    "(looked for 'media_refs'). Make sure that custom node "
                    "pack is installed and enabled, then restart ComfyUI."
                )
        if self.fr_core is None:
            self.fr_core = _find_module("check_bundle")
            if self.fr_core is None:
                raise RuntimeError(
                    "Combined MiniMax H3 node: could not find the Fantastic "
                    "pack's refmod_core module (looked for 'check_bundle')."
                )
        if self.fr_mods is None:
            self.fr_mods = _find_module("KIND_LABEL")
            if self.fr_mods is None:
                raise RuntimeError(
                    "Combined MiniMax H3 node: could not find the Fantastic "
                    "pack's refmods module (looked for 'KIND_LABEL')."
                )
        return self.uc, self.fr_nodes, self.fr_core, self.fr_mods


_deps = _Deps()


class PK_MiniMaxH3Combined(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PK_MiniMaxH3Combined",
            display_name="PK MiniMax H3 Combined Image/Reference to Video",
            category=CATEGORY,
            description=(
                "Combines UtilsCollection's Advanced MiniMax H3 Image to Video "
                "(keyframes, native reference images, reference video/audio) with "
                "the Fantastic Prompt Builder's Media Loader bundle and RefMod "
                "bundle, in one prompt encode. Visual fusion, media timestamp "
                "config, clip continuation and temporal fusion are NOT included "
                "here — use UC's own node directly for those."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="MiniMax H3 Qwen3-VL 32B text encoder (qwen3vl_32b)."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped to H3's 17k+5 grid."),
                io.Vae.Input("vae", optional=True,
                    tooltip="H3 video VAE. Needed for keyframes, native reference "
                            "images/video, Media Loader visuals, and RefMod visuals."),
                io.Vae.Input("audio_vae", optional=True,
                    tooltip="H3 audio VAE. Needed for reference audio / voices."),
                io.Image.Input("first_frame", optional=True,
                    tooltip="Frame-zero VAE anchor. Mutually exclusive with "
                            "reference_images (native H3 rule)."),
                io.Image.Input("last_frame", optional=True,
                    tooltip="Final-frame VAE anchor. Mutually exclusive with "
                            "reference_images (native H3 rule)."),
                io.Autogrow.Input(
                    "reference_images",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image"),
                        prefix="reference_image_",
                        min=0,
                        max=32,  # raise if your build of the native H3 node allows more
                    ),
                    tooltip="Native H3 reference image(s), one per slot — auto-expands "
                            "and collapses like the native H3 node's reference_images "
                            "input. Mutually exclusive with first_frame/last_frame.",
                ),
                io.Image.Input("video", optional=True,
                    tooltip="A single reference video (frame batch, 24 fps)."),
                io.Audio.Input("audio", optional=True,
                    tooltip="A single standalone reference audio clip."),
                io.Combo.Input("ref_image_size", options=["match", "max", "none"], default="match",
                    tooltip="match: scale native references to the generation's pixel area. "
                            "max: keep up to a 2048px short edge. none: text-encoder only, no VAE encode."),
                io.Int.Input("vlm_resolution", default=384, min=0, max=4096, step=32,
                    tooltip="Qwen3-VL presentation resolution for still pictures (keyframes and "
                            "reference images). 0 (or out of 256-4096) keeps original size."),
                io.Int.Input("vlm_video_resolution", default=384, min=0, max=4096, step=32,
                    tooltip="Qwen3-VL presentation resolution for the reference `video` input's "
                            "sampled frames. Higher values use more visual tokens. 0 (or out of "
                            "256-4096) keeps original size."),
                io.Custom("H3_REFS").Input("references", optional=True,
                    tooltip="Media Loader bundle (from the Fantastic pack's "
                            "MiniMax H3 Media Loader, or its Prompt Builder's "
                            "'references' output). Labelled after the inputs "
                            "above, before RefMods."),
                io.Custom("H3_REF_MODS").Input("mods", optional=True,
                    tooltip="RefMod bundle from a RefMod Stack, or the Prompt "
                            "Builder's 'mods' output. Labelled last."),
                io.Float.Input("reference_fps", default=24.0, min=1.0, max=120.0,
                    tooltip="Playback rate assumed when reconstructing a RefMod video reference."),
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
                references=None, mods=None, reference_fps=24.0, max_total_tokens=0) -> io.NodeOutput:
        uc, fr_nodes, fr_core, fr_mods = _deps.resolve()

        if not uc.is_minimax_h3_text_encoder(clip):
            raise ValueError("Combined MiniMax H3 node requires the qwen3vl_32b (H3) text encoder.")
        if not math.isfinite(reference_fps) or not 1 <= reference_fps <= 120:
            raise ValueError("reference_fps must be between 1 and 120.")

        # `reference_images` is an Autogrow group: a dict of
        # {"reference_image_0": tensor, "reference_image_1": tensor, ...} in
        # slot order, one entry per connected slot (each entry may itself be
        # a batch). Pass the whole dict through so flattening stays ordered
        # and per-slot, instead of the old single-batch collapse.
        flat_refs = []
        if reference_images:
            _, flat_refs, _ = uc.extract_and_flatten_images(reference_images)

        keyframe_mode = first_frame is not None or last_frame is not None
        native_mode = bool(flat_refs)
        if keyframe_mode and native_mode:
            raise ValueError(
                "MiniMax H3 frame inputs (first_frame/last_frame) cannot be combined with "
                "native reference_images (same rule as UC's own node)."
            )
        if keyframe_mode and video is not None:
            raise ValueError("MiniMax H3 frame inputs cannot be used together with a reference video.")

        latent, frame_count = uc.minimax_h3_empty_av_latent(width, height, length)
        frame_vae_enabled = ref_image_size != "none"

        counters = {"image": 0, "video": 0, "audio": 0}
        vlm_items = []      # Qwen/VLM presentation items -> minimax_ref_items (or images= for the plain case)
        keyframes = []      # -> metadata["minimax_keyframes"]
        refs = []            # -> metadata["minimax_refs"]
        mapping = []

        # ---- keyframes: embed into the joint latent at frame 0 / last frame ----
        if first_frame is not None:
            vlm_items.append(uc.prepare_vlm_image(first_frame, vlm_resolution))
            counters["image"] += 1
            mapping.append(f"<Picture {counters['image']}> = first frame")
            if vae is not None and frame_vae_enabled:
                px = uc.prepare_minimax_h3_frame(first_frame, width, height, "disabled")
                keyframes.append({"resolved_frame_index": 0, "latent": vae.encode(px)})
        if last_frame is not None:
            vlm_items.append(uc.prepare_vlm_image(last_frame, vlm_resolution))
            counters["image"] += 1
            mapping.append(f"<Picture {counters['image']}> = last frame")
            if vae is not None and frame_vae_enabled:
                px = uc.prepare_minimax_h3_frame(last_frame, width, height, "center")
                keyframes.append({"resolved_frame_index": frame_count - 1, "latent": vae.encode(px)})

        # ---- native H3 reference images: separate reference blocks ----
        for n, img in enumerate(flat_refs, 1):
            vlm_items.append(uc.prepare_vlm_image(img, vlm_resolution))
            counters["image"] += 1
            mapping.append(f"<Picture {counters['image']}> = reference image {n}")
            if vae is not None and frame_vae_enabled:
                px = uc.prepare_minimax_h3_reference_image(img, width, height, ref_image_size)
                th, tw = px.shape[1], px.shape[2]
                refs.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": vae.encode(px)})

        # ---- reference video ----
        if video is not None:
            if vae is None:
                raise ValueError("`video` needs the H3 video VAE.")
            frames, block = uc.prepare_minimax_h3_reference_video(video, vae, frame_count, encode_reference=True)
            counters["video"] += 1
            mapping.append(f"<Video {counters['video']}> = video")
            sample_idx = list(range(0, frames.shape[0], 12))
            vlm_video = uc.prepare_minimax_h3_vlm_video_frames(frames[sample_idx], vlm_video_resolution)
            vlm_items.append(("video", vlm_video, [i / 24.0 for i in sample_idx]))
            refs.append(block)

        # ---- standalone reference audio ----
        if audio is not None:
            if audio_vae is None:
                raise ValueError("`audio` needs audio_vae.")
            block = uc._encode_minimax_h3_audio_reference(audio, audio_vae)
            counters["audio"] += 1
            mapping.append(f"<Audio {counters['audio']}> = audio")
            vlm_items.append(("audio", None, None))
            refs.append(block)

        # ---- Media Loader bundle (Fantastic pack) ----
        if references is not None:
            m_items, m_blocks, m_mapping = fr_nodes.media_refs(
                references, vae, audio_vae, ref_image_size, width, height, length, counters)
            for it in m_items:
                if it["type"] == "image":
                    vlm_items.append(it["data"])
                elif it["type"] == "video":
                    vlm_items.append(("video", it["data"], it.get("timestamps")))
                else:
                    vlm_items.append(("audio", None, None))
            refs.extend(m_blocks)
            mapping.extend(m_mapping)

        # ---- RefMod bundle (shared with Luisa's ComfyUI-MiniMaxH3Mod) ----
        active = [(m, s) for m, s in fr_core.check_bundle(mods, "Combined MiniMax H3 node") if s > 0]
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
            mapping.append(f"<{fr_mods.KIND_LABEL[kind]} {counters[kind]}> = {mod.name}")
            if kind == "audio":
                vlm_items.append(("audio", None, None))
            else:
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
                    vlm_items.append(pixels[:1].clone())
                else:
                    times = [i / 2 for i in range(math.ceil(pixels.shape[0] * 2 / reference_fps))]
                    idx = [min(round(t * reference_fps), pixels.shape[0] - 1) for t in times]
                    vlm_items.append(("video", pixels[idx].clone(), times))
            refs.append(block)
        del decoded

        # ---- tokenize + encode ----
        # Native H3 rule (matches UC's own node): plain keyframe-only / T2VA
        # prompts use images=[...]; anything involving native references,
        # video, audio, Media Loader refs, or RefMods switches to the richer
        # minimax_ref_items=[...] contract.
        use_ref_items = bool(native_mode or video is not None or audio is not None
                              or references is not None or active)
        if use_ref_items:
            reference_items = []
            for entry in vlm_items:
                if isinstance(entry, tuple):
                    kind, data, timestamps = entry
                    item = {"type": kind}
                    if data is not None:
                        item["data"] = data
                    if timestamps is not None:
                        item["timestamps"] = timestamps
                    reference_items.append(item)
                else:
                    reference_items.append({"type": "image", "data": entry})
            tokens = clip.tokenize(prompt, minimax_ref_items=reference_items)
        else:
            images = [entry for entry in vlm_items if not isinstance(entry, tuple)]
            tokens = clip.tokenize(prompt, images=images)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        out = []
        for embedding, metadata in conditioning:
            metadata = dict(metadata)
            if keyframes:
                metadata["minimax_keyframes"] = list(metadata.get("minimax_keyframes", [])) + keyframes
                metadata["minimax_frame_count"] = frame_count
            if refs:
                metadata["minimax_refs"] = list(metadata.get("minimax_refs", [])) + refs
            out.append([embedding, metadata])

        return io.NodeOutput(out, latent, "\n".join(mapping) or "No references.")


class PK_MiniMaxH3CombinedExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PK_MiniMaxH3Combined]


async def comfy_entrypoint() -> PK_MiniMaxH3CombinedExtension:
    return PK_MiniMaxH3CombinedExtension()
