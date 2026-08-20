"""ModelPatcher surgery for the AdaLN LoRA fix.

``adaln.py`` owns the maths and where the basis comes from. This module owns the
ComfyUI side: find the AdaLN patches that cannot be applied to a curve-form base,
drop them, and (in port mode) put back an equivalent pair of ``("diff",)`` patches
on the weight and its bias.

Detection is by shape, so nothing has to raise before we act. The patches we emit
are ordinary ComfyUI patches -- no forward hooks, no per-step cost.
"""

from __future__ import annotations

import logging
import uuid

import torch

from comfy.weight_adapter import LoRAAdapter

from .adaln import (ADALN_WEIGHT_SUFFIX, BASIS_SUBDIR, GRID_FILENAME,
                    get_basis, get_basis_dense, grid_from_time_embedder,
                    pinv_basis, port_delta, port_factors_reverse)

log = logging.getLogger("H3AdaLN")

# Above this relative residual the affine fit is not describing the same curve the
# checkpoint was pruned from, so porting would inject noise. Strip instead.
MAX_RESIDUAL = 5e-3


def _lora_weights(value):
    """The ``weights`` tuple of a plain LoRA adapter, or None for anything else.

    Anything that is not a 2-D LoRA -- a ``("diff",)`` tuple, LoHa/LoKr, a conv
    adapter -- is not ours to touch and is left exactly as it is.
    """
    weights = getattr(value, "weights", None)
    if weights is None or len(weights) < 3:
        return None
    up, down = weights[0], weights[1]
    if not (torch.is_tensor(up) and torch.is_tensor(down)):
        return None
    if up.ndim != 2 or down.ndim != 2:
        return None
    return weights


def _fits(weights, target):
    """True when ComfyUI can already reshape this delta onto the target weight."""
    return (weights[0].shape[0] == target.shape[0]
            and weights[1].shape[1] == target.shape[1])


def _unportable_reason(patch, weights, target, basis_v, reverse):
    """Why this mismatched patch cannot be rebased, or None if it can be.

    ``basis_v`` is ``[D, k]``. Forward (dense LoRA -> pruned base) needs a ``D``-wide
    LoRA and a ``k``-wide target; reverse (curve LoRA -> dense base) needs the
    opposite.
    """
    if len(weights) > 3 and weights[3] is not None:
        return "locon mid weights"
    if len(weights) > 4 and weights[4] is not None:
        return "DoRA scale"
    if len(weights) > 5 and weights[5] is not None:
        return "reshape_weight"
    if patch[2] != 1.0:
        return "strength_model=%r scales the base weight" % (patch[2],)
    if patch[3] is not None:
        return "patch offset"
    if patch[4] is not None:
        return "patch function"

    want_lora, want_target = (basis_v.shape[1], basis_v.shape[0]) if reverse else \
        (basis_v.shape[0], basis_v.shape[1])
    if weights[1].shape[1] != want_lora:
        return ("LoRA input dim %d, basis expects %d"
                % (weights[1].shape[1], want_lora))
    if target.shape[1] != want_target:
        return ("target width %d, basis produces %d"
                % (target.shape[1], want_target))
    if weights[0].shape[0] != target.shape[0]:
        return ("LoRA output dim %d, target %d"
                % (weights[0].shape[0], target.shape[0]))
    return None


def patch_scale(patch, weights):
    """Effective multiplier, matching comfy/weight_adapter/lora.py:248-251."""
    alpha = weights[2]
    rank = weights[1].shape[0]
    return float(patch[0]) * (float(alpha) / rank if alpha is not None else 1.0)


def scan_mismatched(patches, state_dict):
    """``{key: (keep, mismatched, target)}`` for AdaLN keys that have a problem."""
    found = {}
    for key in list(patches.keys()):
        if not key.endswith(ADALN_WEIGHT_SUFFIX):
            continue
        target = state_dict.get(key)
        if target is None or target.ndim != 2:
            continue
        keep, bad = [], []
        for patch in patches[key]:
            weights = _lora_weights(patch[1])
            if weights is None or _fits(weights, target):
                keep.append(patch)
            else:
                bad.append((patch, weights))
        if bad:
            found[key] = (keep, bad, target)
    return found


def _store_dtype(tensor):
    """Store deltas at the target dtype, which is what ComfyUI casts to anyway."""
    dtype = getattr(tensor, "dtype", None)
    return dtype if getattr(dtype, "is_floating_point", False) else torch.float32


def _check(basis, missing):
    if basis is None:
        return None, missing
    residual = basis[2]
    if residual != residual or residual > MAX_RESIDUAL:      # NaN-safe
        return None, ("best basis fit is %.2e (limit %.0e), too poor to trust"
                      % (residual, MAX_RESIDUAL))
    return basis, None


def resolve_basis(diffusion_model):
    """``(basis, None)`` for a pruned base, or ``(None, reason)``.

    The table is the model own; the dense curve has to be found.
    """
    table = getattr(diffusion_model, "adaln_t_table", None)
    if not torch.is_tensor(table) or table.ndim != 2:
        return None, "model has no readable adaln_t_table"
    if table.device.type == "meta":
        return None, "adaln_t_table is not materialised"
    try:
        basis = get_basis(table)
    except Exception as exc:                             # noqa: BLE001
        log.debug("[H3AdaLN] basis lookup failed", exc_info=True)
        return None, "basis lookup failed: %s" % (exc,)
    return _check(basis, "no AdaLN basis available: put a non-pruned H3 checkpoint "
                         "in models/diffusion_models, or a %s in models/%s"
                         % (GRID_FILENAME, BASIS_SUBDIR))


def resolve_basis_reverse(diffusion_model):
    """``(basis, None)`` for a dense base, or ``(None, reason)``.

    The mirror situation: the dense curve is known exactly -- it is this model own
    time embedder -- and what has to be found is the table, i.e. which pruned build
    the curve LoRA was trained against.
    """
    grid = None
    embedder = getattr(diffusion_model, "time_embedder", None)
    if embedder is not None:
        grid = grid_from_time_embedder(embedder)
    if grid is None:
        return None, ("could not read this model time embedder, so the AdaLN curve "
                      "is unknown")
    try:
        basis = get_basis_dense(grid)
    except Exception as exc:                             # noqa: BLE001
        log.debug("[H3AdaLN] reverse basis lookup failed", exc_info=True)
        return None, "basis lookup failed: %s" % (exc,)
    return _check(basis, "no adaln_t_table found: put a pruned (curve-form) H3 "
                         "checkpoint in models/diffusion_models")


def fix_model(model, mode="port"):
    """Return ``(model, report)`` with the dense AdaLN LoRA ported or stripped.

    ``mode`` is ``port``, ``strip`` or ``off``. ``port`` degrades to ``strip`` --
    with the reason recorded in the report -- whenever no trustworthy basis exists.
    """
    report = {"mode": mode, "effective_mode": mode, "status": "ok", "keys": 0,
              "ported": 0, "stripped": 0, "unportable": 0, "residual": None,
              "source": None, "direction": None, "notes": []}

    if mode == "off":
        report["status"] = "disabled"
        return model, report

    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        report["status"] = "not a diffusion model, nothing to do"
        return model, report

    # Which way the LoRA has to be rebased is decided by the base, not the LoRA: a
    # pruned base takes dense LoRAs and needs dense->curve; a dense base takes
    # curve-form LoRAs and needs the reverse.
    reverse = not getattr(diffusion_model, "use_adaln_curves", False)
    if reverse and not hasattr(diffusion_model, "time_embedder"):
        report["status"] = "not a MiniMax-H3 model, nothing to do"
        return model, report
    report["direction"] = "curve->dense" if reverse else "dense->curve"

    found = scan_mismatched(model.patches, model.model_state_dict())
    if not found:
        report["status"] = "no mismatched AdaLN patches"
        return model, report
    report["keys"] = len(found)

    basis = None
    if mode == "port":
        basis, reason = (resolve_basis_reverse(diffusion_model) if reverse
                         else resolve_basis(diffusion_model))
        if basis is None:
            report["effective_mode"] = "strip"
            report["notes"].append(reason)
        else:
            report["residual"], report["source"] = basis[2], basis[3]

    patched = model.clone()
    state_dict = patched.model_state_dict()
    new_patches = {}
    pinv_v = pinv_basis(basis[1]) if (reverse and basis is not None) else None

    for key, (keep, bad, target) in found.items():
        if keep:
            patched.patches[key] = keep
        else:
            patched.patches.pop(key, None)

        if basis is None:
            report["stripped"] += len(bad)
            continue

        c, basis_v = basis[0], basis[1]
        delta_w = delta_b = None
        ups, downs = [], []           # reverse only: low-rank factors to concatenate
        for patch, weights in bad:
            reason = _unportable_reason(patch, weights, target, basis_v, reverse)
            if reason is not None:
                report["unportable"] += 1
                report["notes"].append("%s: %s" % (key, reason))
                continue
            scale = patch_scale(patch, weights)
            if reverse:
                # Stays factored: the multiplied-out dense delta would be ~1 GB.
                a_dense, a_c = port_factors_reverse(weights[1], scale, c, pinv_v)
                ups.append(weights[0])
                downs.append(a_dense.to(weights[0].dtype))
                step_b = -(weights[0].detach().float() @ a_c)
            else:
                step_w, step_b = port_delta(weights[0], weights[1], scale, c, basis_v)
                delta_w = step_w if delta_w is None else delta_w + step_w
            delta_b = step_b if delta_b is None else delta_b + step_b
            report["ported"] += 1

        if reverse and ups:
            # Concatenating the factors sums the deltas exactly, keeping one
            # adapter per key however many LoRAs are stacked on it.
            merged = (torch.cat(ups, dim=1), torch.cat(downs, dim=0),
                      None, None, None, None)
            new_patches[key] = LoRAAdapter(set(), merged)
        elif not reverse and delta_w is not None:
            new_patches[key] = ("diff", (delta_w.to(_store_dtype(target)),))
        else:
            continue

        bias_key = key[:-len("weight")] + "bias"
        bias = state_dict.get(bias_key)
        if bias is None:
            report["notes"].append(
                "%s missing, DC term of the port dropped" % bias_key)
        else:
            new_patches[bias_key] = ("diff", (delta_b.to(_store_dtype(bias)),))

    if new_patches:
        patched.add_patches(new_patches, 1.0)
    patched.patches_uuid = uuid.uuid4()
    return patched, report


def format_report(report):
    """One log line. Never one line per key -- that is the bug being fixed."""
    if report["status"] != "ok":
        return "AdaLN LoRA fix: %s" % report["status"]
    if report["effective_mode"] == "strip":
        line = ("AdaLN LoRA fix: stripped %d incompatible patch(es) across %d key(s)"
                % (report["stripped"], report["keys"]))
    else:
        line = ("AdaLN LoRA fix: ported %d patch(es) across %d key(s) %s, basis "
                "from %s (residual %.2e)"
                % (report["ported"], report["keys"], report["direction"],
                   report["source"], report["residual"]))
    if report["unportable"]:
        line += ", %d could not be ported" % report["unportable"]
    return line
