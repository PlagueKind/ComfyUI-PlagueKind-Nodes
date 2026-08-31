import os
import sys
import asyncio
import importlib.util

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

current_dir = os.path.dirname(__file__)

# Sub-packages that ship their own __init__.py using ComfyUI's V3
# comfy_entrypoint/ComfyExtension API. They keep working internal relative
# imports, so they're loaded as real packages and merged into this pack's
# V1-style mappings below, instead of being flattened by the generic
# per-file walker (which has no concept of packages or comfy_entrypoint).
EXTENSION_PACKAGES = [
    "ComfyUI-H3-AdaLN-LoRA-Fix",
    "ComfyUI-H3-SLA-Attention",
    "ComfyUI-H3-UltimateUpscale-LD",
]


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # A loop is already running in this thread (e.g. ComfyUI's own
        # server loop mid-startup) - run it in a fresh thread instead.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            return ex.submit(asyncio.run, coro).result()


def _load_extension_package(pkg_dir_name):
    pkg_path = os.path.join(current_dir, pkg_dir_name)
    init_path = os.path.join(pkg_path, "__init__.py")
    if not os.path.isfile(init_path):
        return
    module_name = "plaguekind_nodes_ext_" + pkg_dir_name.replace("-", "_")
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, init_path, submodule_search_locations=[pkg_path])
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        entrypoint = getattr(module, "comfy_entrypoint", None)
        if entrypoint is None:
            return  # package logged its own failure and disabled itself

        extension = _run_async(entrypoint())
        node_classes = _run_async(extension.get_node_list())
        for cls in node_classes:
            schema = cls.define_schema()
            NODE_CLASS_MAPPINGS[schema.node_id] = cls
            NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = (
                schema.display_name or schema.node_id)
    except Exception as e:
        print(f"[PlagueKind-Nodes] Failed to load extension package {pkg_dir_name}: {e}")


for pkg_dir_name in EXTENSION_PACKAGES:
    _load_extension_package(pkg_dir_name)

_extension_dirs = {os.path.join(current_dir, p) for p in EXTENSION_PACKAGES}

for root, dirs, files in os.walk(current_dir):
    if root == current_dir:
        continue
    if any(root == d or root.startswith(d + os.sep) for d in _extension_dirs):
        continue  # handled above as real packages
    if root not in sys.path:
        sys.path.insert(0, root)
    for file in files:
        if file.endswith(".py") and not file.startswith("_"):
            file_path = os.path.join(root, file)
            module_name = file[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    if hasattr(module, "NODE_CLASS_MAPPINGS"):
                        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"[PlagueKind-Nodes] Failed to load {file} from {root}: {e}")

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
