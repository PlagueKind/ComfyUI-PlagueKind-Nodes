"""End-to-end check for the H3 AdaLN LoRA Fix.

Real LoRA -> real ComfyUI lora pipeline -> real calculate_weight.

Builds only the AdaLN projections at their true shapes (~90 MB), so the exact code
path that emits `ERROR lora ...` runs for real without loading a 20 GB checkpoint.
"""
import importlib.util, logging, os, sys

import torch
import safetensors

PKG = os.path.dirname(os.path.abspath(__file__))
# .../ComfyUI/custom_nodes/<this pack>  ->  .../ComfyUI
COMFY = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY)
import comfy.lora
import comfy.lora_convert
import comfy.utils
from comfy.model_patcher import ModelPatcher

spec = importlib.util.spec_from_file_location(
    "h3u", os.path.join(PKG, "__init__.py"), submodule_search_locations=[PKG])
h3u = importlib.util.module_from_spec(spec)
sys.modules["h3u"] = h3u
spec.loader.exec_module(h3u)
from h3u import adaln_patch

def _header(path):
    import json, struct
    with open(path, "rb") as fh:
        return json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))


def _find(root, pick, label):
    """First file under *root* that `pick(header)` accepts."""
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(dirpath, name)
            try:
                if pick(_header(path)):
                    return path
            except Exception:                                 # noqa: BLE001
                continue
    raise SystemExit(
        "could not find %s under %s -- edit MODEL/LORA at the top of this file"
        % (label, root))


MODEL = os.environ.get("H3_MODEL") or _find(
    os.path.join(COMFY, "models", "diffusion_models"),
    lambda h: any(k.endswith("adaln_t_table") for k in h),
    "a pruned (curve-form) H3 checkpoint")

LORA = os.environ.get("H3_LORA") or _find(
    os.path.join(COMFY, "models", "loras"),
    lambda h: any(k.startswith("diffusion_model.") for k in h) and any(
        "adaln_proj" in k and k.endswith("lora_A.weight") and h[k]["shape"][1] == 2688
        for k in h),
    "a dense-AdaLN H3 LoRA with diffusion_model.* keys")

print("model: %s" % os.path.basename(MODEL))
print("lora : %s\n" % os.path.basename(LORA))


class Errors(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class Proj(torch.nn.Module):
    def __init__(self, out, k):
        super().__init__()
        self.linear = torch.nn.Linear(k, out, bias=True, dtype=torch.float16)


class Blk(torch.nn.Module):
    def __init__(self, out, k):
        super().__init__()
        self.adaln_proj = Proj(out, k)


class DiT(torch.nn.Module):
    def __init__(self, table, n=50):
        super().__init__()
        k = table.shape[1]
        self.blocks = torch.nn.ModuleList([Blk(96768, k) for _ in range(n)])
        self.final_layer = Blk(10752, k)
        self.register_buffer("adaln_t_table", table)
        self.use_adaln_curves = True


class Wrapper(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.diffusion_model = dit


with safetensors.safe_open(MODEL, "pt") as f:
    table = f.get_tensor("adaln_t_table").float()

model = ModelPatcher(Wrapper(DiT(table)), torch.device("cpu"), torch.device("cpu"))
sd_keys = set(model.model_state_dict().keys())
print("built %d adaln weights, e.g. %s" % (
    sum(1 for k in sd_keys if k.endswith("adaln_proj.linear.weight")),
    sorted(k for k in sd_keys if k.endswith("adaln_proj.linear.weight"))[0]))

lora = comfy.utils.load_torch_file(LORA, safe_load=True)
lora = comfy.lora_convert.convert_lora(lora)
key_map = {}
for k in sd_keys:
    if k.startswith("diffusion_model.") and k.endswith(".weight"):
        key_map[k[:-len(".weight")]] = k
loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
adaln_loaded = {k: v for k, v in loaded.items() if "adaln_proj" in k}
print("lora keys mapped onto adaln projections: %d" % len(adaln_loaded))
model.add_patches(adaln_loaded, 1.0)

probe = sorted(k for k in model.patches if k.endswith("adaln_proj.linear.weight"))[0]


def apply(patcher, key):
    handler = Errors()
    logging.getLogger().addHandler(handler)
    try:
        base = patcher.model_state_dict()[key].to(torch.float32).clone()
        out = comfy.lora.calculate_weight(patcher.patches.get(key, []), base, key)
    finally:
        logging.getLogger().removeHandler(handler)
    return out, handler.lines


base_w = model.model_state_dict()[probe].to(torch.float32).clone()

print("\n--- BEFORE the fix (current behaviour) ---")
out, errs = apply(model, probe)
print("errors logged : %d" % len(errs))
if errs:
    print("  %s" % errs[0])
print("weight changed: %s" % (not torch.equal(out, base_w)))

for mode in ("strip", "port"):
    fixed, report = adaln_patch.fix_model(model, mode)
    print("\n--- AFTER fix, mode=%s ---" % mode)
    print(adaln_patch.format_report(report))
    out, errs = apply(fixed, probe)
    print("errors logged : %d" % len(errs))
    changed = not torch.equal(out, base_w)
    print("weight changed: %s" % changed)
    if changed:
        delta = (out - base_w)
        print("delta norm    : %.4f  (base norm %.4f)" % (delta.norm(), base_w.norm()))
    bias_key = probe[:-len("weight")] + "bias"
    print("bias patched  : %s" % (bias_key in fixed.patches))
