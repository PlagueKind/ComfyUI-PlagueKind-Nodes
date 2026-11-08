"""PK Media Loader - 9 image + 3 video file slots in one node.

Loads stills and videos straight from ComfyUI's input folder - the core
Load Image + Load Video nodes, combined into a single node:

  * image_1..image_9 - image file pickers (uploadable), loaded with the
                       core Load Image class itself (pyav + Pillow fallback,
                       identical tensors to the stock node).
  * video_1..video_3 - video file pickers (uploadable), decoded to frame
                       batches (B, H, W, 3) through the same pyav path the
                       core video stack uses (InputImpl.VideoFromFile).
  * video_1_audio..video_3_audio - per-video toggles (default on): when
                       off, that video's audio channel is left empty.

Outputs:
  * bundle  - a BUNDLE dict keyed input_1..input_15 (15 channels). Drops
              directly into Bundle Out (PlagueKind), whose output_i carries
              input_i positionally, exactly like Bundle In:
                input_1..input_9    image slots 1-9
                input_10..input_12  video slots 1-3, frame batches (IMAGE)
                input_13..input_15  video slots 1-3, audio (AUDIO dicts;
                                    empty when the file has no audio track)
  * image_1 - image slot 1 as a plain IMAGE tensor, so the resize chain can
              stay wired without routing through Bundle Out.

Slots left on "(none)" are skipped (their bundle slot stays empty, no
error). A file that can't be read warns to the console and skips instead of
killing the whole run.
"""

import os

import folder_paths
import comfy.model_management
import nodes  # ComfyUI core - LoadImage does the heavy lifting for stills

from comfy_api.latest import ComfyExtension, InputImpl, io

CATEGORY = "PlagueKind/loaders"
NONE = "(none)"
NUM_IMAGES = 9
NUM_VIDEOS = 3


class PK_MediaLoader(io.ComfyNode):
    @classmethod
    def _file_list(cls, content_type):
        input_dir = folder_paths.get_input_directory()
        try:
            files = [f for f in os.listdir(input_dir)
                     if os.path.isfile(os.path.join(input_dir, f))]
        except Exception:
            files = []
        files = folder_paths.filter_files_content_types(files, [content_type])
        return [NONE] + sorted(files)

    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = []
        for i in range(NUM_IMAGES):
            inputs.append(io.Combo.Input(
                f"image_{i + 1}", options=cls._file_list("image"), default=NONE,
                optional=True, upload=io.UploadType.image,
                tooltip=f"Image {i + 1} (leave on (none) to skip). "
                        f"image_1 also feeds the extra image_1 output."))
        for i in range(NUM_VIDEOS):
            inputs.append(io.Combo.Input(
                f"video_{i + 1}", options=cls._file_list("video"), default=NONE,
                optional=True, upload=io.UploadType.video,
                tooltip=f"Video {i + 1} (leave on (none) to skip) - loaded as a "
                        f"frame batch (bundle input_{NUM_IMAGES + i + 1}) plus its "
                        f"audio track (bundle input_{NUM_IMAGES + NUM_VIDEOS + i + 1}, "
                        "empty when the file has no audio)."))
        # Per-video audio toggles - appended after all file pickers so
        # pre-toggle workflows keep their positional widget values (the new
        # widgets simply pick up their defaults). optional=True keeps them in
        # the optional group, which the frontend orders after the required
        # group - so they render at the end of the list, after the videos.
        for i in range(NUM_VIDEOS):
            inputs.append(io.Boolean.Input(
                f"video_{i + 1}_audio", default=True, optional=True,
                tooltip=f"Output audio for video {i + 1} (bundle "
                        f"input_{NUM_IMAGES + NUM_VIDEOS + i + 1}). Uncheck to "
                        "leave that channel empty."))
        return io.Schema(
            node_id="PK_MediaLoader",
            display_name="PK Media Loader ( 9 Images + 3 Videos )",
            category=CATEGORY,
            description=(
                "Nine image + three video file pickers in one node (all slots "
                "optional). Emits a BUNDLE (input_1..input_15: images 1-9, "
                "video frames 1-3, video audio 1-3) for Bundle Out, plus "
                "image slot 1 as a plain IMAGE."
            ),
            inputs=inputs,
            outputs=[
                io.Custom("BUNDLE").Output(id="bundle",
                    tooltip="input_1..input_15: images 1-9, video frame "
                            "batches 1-3 (input_10-12), video audio 1-3 "
                            "(input_13-15). Feed into Bundle Out (PlagueKind)."),
                io.Image.Output(id="image_1",
                    tooltip="Image slot 1 as a plain IMAGE tensor (for the resize chain)."),
            ],
        )

    # -- loading ----------------------------------------------------------

    @classmethod
    def _load_image(cls, name):
        """Core Load Image semantics for one (possibly annotated) file name."""
        images, _mask = nodes.LoadImage().load_image(name)
        return images

    @classmethod
    def _load_video(cls, name):
        path = folder_paths.get_annotated_filepath(name)
        components = InputImpl.VideoFromFile(path).get_components()
        images = components.images
        # AUDIO dict {"waveform": (1, C, T), "sample_rate": int} or None when
        # the file has no audio track - the exact shape the core audio nodes
        # produce, so it drops into any AUDIO input downstream.
        audio = components.audio
        return (
            images.to(device=comfy.model_management.intermediate_device(),
                      dtype=comfy.model_management.intermediate_dtype()),
            audio,
        )

    @staticmethod
    def _load_or_skip(loader, name, label):
        if not name or name == NONE:
            return None
        try:
            return loader(name)
        except Exception as e:
            print(f"[PlagueKind | PK_MediaLoader] {label} '{name}' failed to load ({e}) - slot left empty.")
            return None

    # -- ComfyUI hooks -----------------------------------------------------

    @classmethod
    def fingerprint_inputs(s, **kwargs):
        """Cache key = slot selections + file mtimes (core LoadVideo style:
        mtime instead of hashing potentially huge video files)."""
        names, stamps = [], []
        for i in range(1, NUM_IMAGES + 1):
            names.append(kwargs.get(f"image_{i}", NONE))
        for i in range(1, NUM_VIDEOS + 1):
            names.append(kwargs.get(f"video_{i}", NONE))
        for i in range(1, NUM_VIDEOS + 1):
            names.append(kwargs.get(f"video_{i}_audio", True))
        for name in names:
            if isinstance(name, bool):  # audio toggles: state is the key
                stamps.append(1 if name else 0)
                continue
            elif not name or name == NONE:
                stamps.append(0)
                continue
            try:
                stamps.append(os.path.getmtime(folder_paths.get_annotated_filepath(name)))
            except Exception:
                stamps.append(-1)
        return tuple(names) + tuple(stamps)

    @classmethod
    def validate_inputs(s, **kwargs):
        for name in list(kwargs.get(f"image_{i}", NONE) for i in range(1, NUM_IMAGES + 1)) \
                + list(kwargs.get(f"video_{i}", NONE) for i in range(1, NUM_VIDEOS + 1)):
            if not name or name == NONE:
                continue
            if not folder_paths.exists_annotated_filepath(name):
                return f"Invalid media file: {name}"
        return True

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        images = []
        for i in range(NUM_IMAGES):
            name = kwargs.get(f"image_{i + 1}", NONE)
            images.append(cls._load_or_skip(cls._load_image, name, f"image_{i + 1}"))
        videos = []
        for i in range(NUM_VIDEOS):
            name = kwargs.get(f"video_{i + 1}", NONE)
            videos.append(cls._load_or_skip(cls._load_video, name, f"video_{i + 1}"))

        # Bundle layout - 15 channels, positional through Bundle Out:
        #   input_1..input_9    image slots 1-9
        #   input_10..input_12  video slots 1-3, frame batches
        #   input_13..input_15  video slots 1-3, audio (empty when silent)
        bundle = {}
        for idx, value in enumerate(images):
            if value is not None:
                bundle[f"input_{idx + 1}"] = value
        for idx, video in enumerate(videos):
            if video is None:
                continue
            frames, audio = video
            if frames is not None:
                bundle[f"input_{NUM_IMAGES + 1 + idx}"] = frames
            if audio is not None and kwargs.get(f"video_{idx + 1}_audio", True):
                bundle[f"input_{NUM_IMAGES + NUM_VIDEOS + 1 + idx}"] = audio
        return io.NodeOutput(bundle, images[0] if images else None)


class PK_MediaLoaderExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PK_MediaLoader]


async def comfy_entrypoint() -> PK_MediaLoaderExtension:
    return PK_MediaLoaderExtension()
