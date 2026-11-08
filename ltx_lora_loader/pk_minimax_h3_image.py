"""PK MiniMax H3 Image - still-image variant of PK MiniMax H3 Combined.

A clone of PK_MiniMaxH3Combined (pk_minimax_h3_combined.py, same directory)
rebuilt around ComfyUI's own `PH3ImageConditioning` (p-h3-image Conditioning)
internals:

* **Sizing matches the native image node.** There is no `ref_image_size`
  option: every reference image is aspect-preserving, down-only, scaled to
  fit the generation's pixel area (core._size_within_area to
  width*height, or 2912x2912 when width/height are 0), exactly like the
  native node. width/height default to 0, meaning "use the first reference
  image's size after scaling, or 2912 without images" - also native behavior.
* **Latent math matches the native image node.** A still image is
  `core._av_latent(latent_width, latent_height, 1, 2)` - one video temporal
  cell (latent_t=1) plus the two audio latents it spans. No `length` input.
* **Video/audio inputs are hidden.** `reference_videos`,
  `reference_video_audios`, `reference_audios`, and `audio_vae` are not
  present (also hidden from the combined node: `first_frame`, `last_frame`,
  and the `references` H3_REFS Media Loader bundle input). The
  vlm_resolution / vlm_video_resolution options are kept: vlm_resolution
  still controls the Qwen3-VL presentation size of reference images, and
  vlm_video_resolution now applies to RefMod video references (the only
  remaining video source in this node).

Everything else - prompt encode, the native-style reference_images autogrow,
the RefMod bundle (mods), reference_fps, max_total_tokens, and the three
outputs - behaves like the combined node. The model needs
ModelSamplingMiniMaxH3 at 1.0/1.0 (same as the native image node).

The shared helpers (VLM resolution, RefMod bundle check, the core-module
loader) are imported from pk_minimax_h3_combined - the pack's walker loads
both files from this directory into sys.path, so the import resolves in
either load order.
"""

import math

import node_helpers
import nodes
from comfy_api.latest import ComfyExtension, io

from pk_minimax_h3_combined import (
    KIND_LABEL,
    _core,
    check_bundle,
    prepare_minimax_h3_vlm_video_frames,
    prepare_vlm_image,
)

CATEGORY = "PlagueKind/minimax"


class PK_MiniMaxH3Image(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PK_MiniMaxH3Image",
            display_name="PK MiniMax H3 Image",
            category=CATEGORY,
            description=(
                "Still-image variant of PK MiniMax H3 Combined, built on the native "
                "p-h3-image internals: reference images sized to the generation's pixel "
                "area (width/height 0 = match the first reference image, or 2912), a "
                "RefMod bundle, and a still-image latent (one video temporal cell + its "
                "two audio latents). The model needs ModelSamplingMiniMaxH3 at 1.0/1.0."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                    tooltip="0 uses the width of the first reference image after scaling, or 2912 without images."),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                    tooltip="0 uses the height of the first reference image after scaling, or 2912 without images."),
                io.Float.Input("reference_fps", default=24.0, min=1.0, max=120.0,
                    tooltip="Playback rate assumed when reconstructing a RefMod video reference's timestamps."),
                io.Int.Input("max_total_tokens", default=0, min=0, max=2147483647,
                    tooltip="Refuse RefMod bundles over this many reference tokens. 0 = no limit."),
                io.Int.Input("vlm_resolution", default=384, min=0, max=4096, step=32,
                    tooltip="Qwen3-VL presentation resolution for still reference images, independent of "
                            "the resolution actually VAE-encoded as the DiT reference. "
                            "0 (or out of 256-4096) keeps the same resolution as the DiT reference."),
                io.Int.Input("vlm_video_resolution", default=384, min=0, max=4096, step=32,
                    tooltip="Qwen3-VL presentation resolution for RefMod video reference frames. "
                            "Higher values use more visual tokens. 0 (or out of 256-4096) keeps original size."),
                io.Vae.Input("vae", optional=True,
                    tooltip="H3 video VAE. Required for reference images to be VAE-encoded as DiT "
                            "references; without it they only condition the text encoder."),
                io.Autogrow.Input("reference_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image",
                            tooltip="Reference image (aspect-preserving, down only, scaled to fit the "
                                    "generation's pixel area)"),
                        prefix="reference_image_", min=0, max=9)),
                io.Custom("H3_REF_MODS").Input("mods", optional=True,
                    tooltip="RefMod bundle from a RefMod Stack, or a Prompt Builder's 'mods' output. Labelled last."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="reference_map"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, width=0, height=0,
                vae=None,
                reference_images=None,
                mods=None,
                reference_fps=24.0, max_total_tokens=0,
                vlm_resolution=384, vlm_video_resolution=384) -> io.NodeOutput:
        core = _core()

        if not math.isfinite(reference_fps) or not 1 <= reference_fps <= 120:
            raise ValueError("reference_fps must be between 1 and 120.")

        # native p-h3-image sizing model: references fit the generation's
        # pixel area; 0x0 means the 2912x2912 image-training area
        area = (width * height) or core.PH3_IMAGE_SIZE ** 2

        ref_items = []          # tokenizer presentation, in order (native refs/RefMods)
        ref_blocks = []          # -> metadata["minimax_refs"], same order
        image_blocks = []        # DiT reference blocks from the reference_images inputs (native's `refs`)
        mapping = []
        counters = {"image": 0, "video": 0, "audio": 0}

        # ---- reference images (Autogrow: reference_image_0, reference_image_1, ...) ----
        # Sizing is the native image node's: aspect-preserving, down only,
        # to the generation's pixel area (the ref_image_size option is gone).
        for img in (reference_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            tw, th = core._size_within_area(w, h, area)
            resized = core._resize_to(img[:1], (tw, th))
            counters["image"] += 1
            mapping.append(f"<Picture {counters['image']}> = reference image")
            ref_items.append({"type": "image", "data": prepare_vlm_image(resized, vlm_resolution)})
            if vae is not None:
                z = vae.encode(resized)
                block = {"kind": "image", "latent_h": z.shape[-2], "latent_w": z.shape[-1], "latent": z}
                image_blocks.append(block)
                ref_blocks.append(block)

        # ---- RefMod bundle ----
        active = [(m, s) for m, s in check_bundle(mods, "MiniMax H3 Image node") if s > 0]
        if max_total_tokens:
            total = sum(m.token_count for m, _s in active)
            if total > max_total_tokens:
                raise ValueError(
                    f"RefMods require {total} tokens; the limit is {max_total_tokens}. "
                    "Lower a weight, drop a pick, or raise the limit.")
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
                    item["data"] = prepare_minimax_h3_vlm_video_frames(pixels[idx].clone(), vlm_video_resolution)
                    item["timestamps"] = times
            ref_items.append(item)
            ref_blocks.append(block)
        del decoded

        # ---- tokenize + encode ----
        # Same convention as the combined node: any reference items ride in
        # minimax_ref_items; with nothing but text it's a plain tokenize.
        if ref_items:
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        else:
            tokens = clip.tokenize(prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if ref_blocks:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"minimax_refs": ref_blocks})

        # native p-h3-image latent: still image (latent_t=1) + the two audio
        # latents it spans. Width/height default to the first reference image's
        # size after scaling, or 2912 without images.
        latent_width = width or (image_blocks[0]["latent_w"] * 16 if image_blocks else core.PH3_IMAGE_SIZE)
        latent_height = height or (image_blocks[0]["latent_h"] * 16 if image_blocks else core.PH3_IMAGE_SIZE)
        latent = core._av_latent(latent_width, latent_height, 1, 2)

        return io.NodeOutput(conditioning, latent, "\n".join(mapping) or "No references.")


class PK_MiniMaxH3ImageExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PK_MiniMaxH3Image]


async def comfy_entrypoint() -> PK_MiniMaxH3ImageExtension:
    return PK_MiniMaxH3ImageExtension()
