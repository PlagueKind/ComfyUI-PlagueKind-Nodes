"""PlagueKind bundle nodes: dynamic multi-input -> single bundle -> multi-output.

Type checking is bypassed via "*" wildcard sockets on both ends; the JS
extension (bundle_nodes.js) handles auto expand/collapse and connected-type
labeling. Mapping between BundleIn's input_i and BundleOut's output_i is
positional, so JS is purely cosmetic — execution is correct even without it.
"""

MAX_SLOTS = 20


class PlagueKindBundleIn:
    CATEGORY = "PlagueKind/utils"
    RETURN_TYPES = ("BUNDLE",)
    RETURN_NAMES = ("bundle",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"input_{i + 1}": ("*", {}) for i in range(MAX_SLOTS)}
        return {"required": {}, "optional": optional}

    def execute(self, **kwargs):
        bundle = {}
        for i in range(MAX_SLOTS):
            key = f"input_{i + 1}"
            if key in kwargs and kwargs[key] is not None:
                bundle[key] = kwargs[key]
        return (bundle,)


class PlagueKindBundleOut:
    CATEGORY = "PlagueKind/utils"
    RETURN_TYPES = tuple("*" for _ in range(MAX_SLOTS))
    RETURN_NAMES = tuple(f"output_{i + 1}" for i in range(MAX_SLOTS))
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bundle": ("BUNDLE", {})}}

    def execute(self, bundle):
        return tuple(bundle.get(f"input_{i + 1}") for i in range(MAX_SLOTS))


NODE_CLASS_MAPPINGS = {
    "PlagueKindBundleIn": PlagueKindBundleIn,
    "PlagueKindBundleOut": PlagueKindBundleOut,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlagueKindBundleIn": "Bundle In (PlagueKind)",
    "PlagueKindBundleOut": "Bundle Out (PlagueKind)",
}
