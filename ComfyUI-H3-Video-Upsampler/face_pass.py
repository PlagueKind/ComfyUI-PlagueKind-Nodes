# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""Small-face re-draw (upsampler issue #3).

Regen2K redraws a face from the latent, and H3's latent is 32x smaller than the
picture: a medium-far face is a few tokens wide, too small for its eyes and
brows to survive the re-draw's noise, so the model draws a plausible face of
its own (heavy brows, eyeliner, dark sockets). Close-ups keep their person,
because their layout spans many tokens.

So every small face gets a close-up of its own. Faces are found in the source
frames (YuNet, OpenCV), linked into tracks and smoothed; each track's face is
cut out of the source with context around it, enlarged so the face is
close-up size, re-drawn by the same upsampler (same LoRA, sigmas, seed,
soundtrack span), then shrunk back and blended over the big re-draw with a
feathered oval. The blend fades out as a face grows past FACE_FULL_PX, so a
face the big re-draw already holds is left to it.

Plug and play: no widget. Without OpenCV or the detector file, it logs once and
the upsampler output is unchanged.
"""
from __future__ import annotations

import hashlib
import math
import os

import torch
import torch.nn.functional as F

TAG = "H3VideoUpsampler"

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

FACE_PASS = True        # False = upsampler output untouched
FACE_FULL_PX = 220      # output face height (px) at or below which the close-up re-draw is used fully
FACE_OFF_PX = 280       # ... and at or above which the big re-draw is kept (was 360 in 0.6.0:
#                         the close-up draws a face ~CROP_OUT / CONTEXT = 358 px, so a 300 px
#                         face gained ~1.2x detail and came out softer than the main re-draw,
#                         issue #3 follow-up; at 280 the close-up still has >= 1.3x)
CONTEXT = 2.5           # crop side = CONTEXT x face height
CROP_IN = 448           # crop edge fed to the upsampler (px)
CROP_OUT = 896          # crop edge the upsampler draws (face ~ CROP_OUT / CONTEXT px tall)
MIN_FACE_SRC = 12       # source face height (px) below which a face is ignored
SCORE = 0.6             # YuNet confidence
MAX_GAP = 12            # frames a track may go undetected (filled by interpolation)
MIN_TRACK = 12          # detected frames a track needs
MAX_TRACKS = 4          # faces re-drawn per clip (largest first)
SMOOTH = 2.0            # box smoothing over time, frames (Gaussian sigma)
EDGE_FADE = 6           # frames a track fades in/out at its ends
MASK_IN, MASK_OUT = 0.55, 0.92   # oval in crop units (face half-height = 1 / CONTEXT)


# ---------------------------------------------------------------- detector

_DET = {}


def _model_path():
    try:
        import folder_paths
        root = os.path.join(folder_paths.models_dir, "facedetection")
    except Exception:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    return os.path.join(root, YUNET_FILE)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ensure_model():
    """Path to the YuNet file, downloaded once (232 KB, MIT). None if unavailable."""
    path = _model_path()
    if os.path.isfile(path) and _sha256(path) == YUNET_SHA256:
        return path
    try:
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = urllib.request.urlopen(YUNET_URL, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != YUNET_SHA256:
            raise RuntimeError("checksum mismatch")
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        print(f"[{TAG}] face detector downloaded to {path}", flush=True)
        return path
    except Exception as exc:
        print(f"[{TAG}] small-face re-draw off: face detector not available ({exc}); "
              f"put {YUNET_FILE} in {os.path.dirname(path)}", flush=True)
        return None


def _detector(w, h):
    try:
        import cv2
    except Exception:
        print(f"[{TAG}] small-face re-draw off: OpenCV (cv2) is not installed", flush=True)
        return None
    path = ensure_model()
    if path is None:
        return None
    key = (path, w, h)
    if key not in _DET:
        _DET.clear()
        _DET[key] = cv2.FaceDetectorYN.create(path, "", (int(w), int(h)), SCORE, 0.3, 50)
    return _DET[key]


def detect(images):
    """[n, H, W, 3] 0..1 -> list per frame of (cx, cy, size) in source px, or None
    when no detector is available. size = face box height."""
    return detect_batches(images[s:s + 16] for s in range(0, int(images.shape[0]), 16))


def detect_batches(batches):
    """detect() over an iterable of [k, H, W, 3] batches (a file read a piece at
    a time). -> list per frame, or None without a detector."""
    import numpy as np
    det, out = None, []
    for batch in batches:
        if det is None:
            det = _detector(int(batch.shape[2]), int(batch.shape[1]))
            if det is None:
                return None
        arr = (batch[..., :3].float().clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        for fr in arr:
            _, faces = det.detect(np.ascontiguousarray(fr[..., ::-1]))
            row = []
            if faces is not None:
                for fa in faces:
                    x, y, fw, fh = (float(v) for v in fa[:4])
                    if fh >= MIN_FACE_SRC:
                        row.append((x + fw / 2.0, y + fh / 2.0, max(fw, fh)))
            out.append(row)
    return out


# ---------------------------------------------------------------- tracks

def _gauss(x, sigma):
    """1-D Gaussian smoothing of a float tensor, edges replicated."""
    if sigma <= 0 or x.numel() < 2:
        return x
    r = max(1, int(3 * sigma + 0.5))
    t = torch.arange(-r, r + 1, dtype=torch.float64)
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = (g / g.sum()).view(1, 1, -1)
    y = F.pad(x.double().view(1, 1, -1), (r, r), mode="replicate")
    return F.conv1d(y, g).view(-1).to(x.dtype)


def link(dets, n):
    """Per-frame detections -> tracks: dicts with first/last frame and per-frame
    (cx, cy, size) tensors over [first, last], gaps filled, smoothed."""
    open_t, done = [], []
    for f in range(n):
        row = list(dets[f]) if f < len(dets) else []
        for tr in sorted(open_t, key=lambda t: -t["pts"][-1][3]):
            px, py, ps = tr["pts"][-1][1:4]
            best, bd = None, None
            for i, (cx, cy, s) in enumerate(row):
                d = math.hypot(cx - px, cy - py)
                if d < 0.6 * max(ps, s) and 1 / 1.6 < s / ps < 1.6 and (bd is None or d < bd):
                    best, bd = i, d
            if best is not None:
                cx, cy, s = row.pop(best)
                tr["pts"].append((f, cx, cy, s))
                tr["last_f"] = f
        for tr in [t for t in open_t if f - t["last_f"] > MAX_GAP]:
            open_t.remove(tr)
            done.append(tr)
        for cx, cy, s in row:
            open_t.append({"pts": [(f, cx, cy, s)], "last_f": f})
    done += open_t
    tracks = []
    for tr in done:
        pts = tr["pts"]
        if len(pts) < MIN_TRACK:
            continue
        f0, f1 = pts[0][0], pts[-1][0]
        fr = torch.tensor([p[0] for p in pts], dtype=torch.float64)
        full = torch.arange(f0, f1 + 1, dtype=torch.float64)
        vals = []
        for k in (1, 2, 3):
            v = torch.tensor([p[k] for p in pts], dtype=torch.float64)
            vals.append(_gauss(_interp(full, fr, v), SMOOTH))
        tracks.append({"first": f0, "last": f1, "cx": vals[0], "cy": vals[1], "size": vals[2]})
    tracks.sort(key=lambda t: -float(t["size"].mean()))
    return tracks


def _interp(x, xp, fp):
    """numpy.interp for 1-D float64 tensors (xp increasing)."""
    idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    w = ((x - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0, 1)
    return y0 + (y1 - y0) * w


def _smoothstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def weights(track, scale, fade_in=True, fade_out=True):
    """Per-frame blend weight of a track's close-up re-draw: 1 for small faces,
    0 past FACE_OFF_PX, faded in/out where the track starts/ends inside the clip
    (a track that starts or ends with the clip is not faded: nothing to hand over to)."""
    s_out = track["size"] * scale
    w = 1.0 - _smoothstep(float(FACE_FULL_PX), float(FACE_OFF_PX), s_out)
    m = w.numel()
    i = torch.arange(m, dtype=torch.float64)
    one = torch.ones(m, dtype=torch.float64)
    up = ((i + 1) / (EDGE_FADE + 1)).clamp(0, 1) if fade_in else one
    down = ((m - i) / (EDGE_FADE + 1)).clamp(0, 1) if fade_out else one
    return (w * torch.minimum(up, down)).float()


# ---------------------------------------------------------------- crop / paste

def crop(images, cx, cy, side, size=CROP_IN, batch=16):
    """Square crops of `images` [n, H, W, 3] around (cx, cy) with edge `side`
    (source px, per frame), resampled to size x size; outside the frame the edge
    pixels repeat. -> [n, size, size, 3]."""
    n, hh, ww = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    out = torch.empty((n, size, size, 3), dtype=torch.float32)
    j = (torch.arange(size, dtype=torch.float64) + 0.5) / size - 0.5       # -0.5..0.5
    for s in range(0, n, batch):
        e = min(n, s + batch)
        x = cx[s:e, None] + j[None, :] * side[s:e, None]                   # source px
        y = cy[s:e, None] + j[None, :] * side[s:e, None]
        gx = (2 * x / ww - 1).float()
        gy = (2 * y / hh - 1).float()
        grid = torch.stack(torch.broadcast_tensors(gx[:, None, :], gy[:, :, None]), dim=-1)
        src = images[s:e, ..., :3].float().permute(0, 3, 1, 2)
        # Anti-alias a crop that shrinks (side > size).
        shrink = float(side[s:e].max()) / size
        if shrink > 1.5:
            k = int(round(shrink)) | 1
            src = F.avg_pool2d(F.pad(src, (k // 2, k // 2, k // 2, k // 2), mode="replicate"),
                               k, stride=1)
        out[s:e] = F.grid_sample(src, grid, mode="bicubic", padding_mode="border",
                                 align_corners=False).permute(0, 2, 3, 1).clamp(0, 1)
    return out


def oval(size):
    """[size, size] blend mask: 1 inside MASK_IN, smooth to 0 at MASK_OUT (crop
    units: the edge is 1, the face's half-height 1 / CONTEXT)."""
    j = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2 - 1
    r = torch.sqrt(j[None, :] ** 2 + (j[:, None] * 0.92) ** 2)
    return 1.0 - _smoothstep(MASK_IN, MASK_OUT, r.double()).float()


def paste(img, redrawn, cx, cy, side, w, sx, sy, dev=None):
    """Blend `redrawn` crops [m, S, S, 3] (float 0..1, or uint8) over `img` [m, Ho, Wo, 3] in place.
    Crop centres/sides are in source px; sx, sy = output px per source px;
    w = per-frame weight. The crop's tone is matched to `img` on the oval's rim."""
    dev = dev or img.device
    m, ho, wo = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
    rims = []
    for t in range(m):
        if float(w[t]) <= 0.0:
            rims.append(None)
            continue
        ccx, ccy, half_x, half_y = cx[t] * sx, cy[t] * sy, side[t] * sx / 2, side[t] * sy / 2
        x0, x1 = max(0, int(math.floor(ccx - half_x))), min(wo, int(math.ceil(ccx + half_x)))
        y0, y1 = max(0, int(math.floor(ccy - half_y))), min(ho, int(math.ceil(ccy + half_y)))
        if x1 <= x0 or y1 <= y0:
            rims.append(None)
            continue
        rims.append((x0, x1, y0, y1, float(ccx), float(ccy), float(half_x), float(half_y)))
    # Tone offset per frame from the rim, smoothed over time. Two passes over the frames,
    # each piece built when it is needed: kept for every frame of a long track, the float
    # pieces were ~20 GB on a 2-minute film (review 11-02). Same pixels either way.
    offs = torch.zeros((m, 3), dtype=torch.float64)
    have = torch.zeros(m, dtype=torch.bool)

    def _piece(t, rim):
        x0, x1, y0, y1, ccx, ccy, hx, hy = rim
        u = (torch.arange(x0, x1, dtype=torch.float32, device=dev) + 0.5 - ccx) / hx
        v = (torch.arange(y0, y1, dtype=torch.float32, device=dev) + 0.5 - ccy) / hy
        grid = torch.stack(torch.broadcast_tensors(u[None, :], v[:, None]), dim=-1)[None]
        tw, th = max(1, x1 - x0), max(1, y1 - y0)
        c = redrawn[t:t + 1].to(dev, torch.float32)
        if redrawn.dtype == torch.uint8:   # 8-bit crops: one frame to float at a time
            c = c / 255.0
        c = c.permute(0, 3, 1, 2)
        side_px = max(2, int(round(2 * max(hx, hy))))
        if side_px < c.shape[-1]:
            c = F.interpolate(c, size=(side_px, side_px), mode="bilinear", antialias=True,
                              align_corners=False)
        piece = F.grid_sample(c, grid, mode="bilinear", padding_mode="border",
                              align_corners=False)[0].permute(1, 2, 0)
        r = torch.sqrt(u[None, :] ** 2 + (v[:, None] * 0.92) ** 2)
        a = 1.0 - _smoothstep(MASK_IN, MASK_OUT, r)
        ring = (r > MASK_IN) & (r < MASK_OUT)
        return piece, a, ring

    for t, rim in enumerate(rims):
        if rim is None:
            continue
        x0, x1, y0, y1 = rim[:4]
        piece, _a, ring = _piece(t, rim)
        base = img[t, y0:y1, x0:x1].to(dev, torch.float32)
        if int(ring.sum()) > 16:
            offs[t] = (base[ring].mean(0) - piece[ring].mean(0)).double().cpu()
            have[t] = True
        del piece, _a, ring, base
    if bool(have.any()):
        idx = torch.nonzero(have).view(-1).double()
        full = torch.arange(m, dtype=torch.float64)
        for ch in range(3):
            v = offs[have, ch]
            offs[:, ch] = (_gauss(_interp(full, idx, v), 3.0) if v.numel() > 1
                           else v.expand(m).clone())
    for t, rim in enumerate(rims):
        if rim is None:
            continue
        x0, x1, y0, y1 = rim[:4]
        piece, a, _ring = _piece(t, rim)
        piece = (piece + offs[t].to(dev, torch.float32)).clamp(0, 1)
        a = (a * float(w[t]))[..., None]
        base = img[t, y0:y1, x0:x1].to(dev, torch.float32)
        img[t, y0:y1, x0:x1] = (base * (1 - a) + piece * a).to(img.device, img.dtype)
    return img


# ---------------------------------------------------------------- driver

ONE_CHUNK_TOKENS = 70000   # the recipe's tile budget: a crop clip this small is re-drawn whole


def one_chunk(m):
    """True when an m-frame crop clip fits one forward at CROP_OUT (no chunk seam)."""
    lat_t = (17 * math.ceil((max(5, m) - 5) / 17) + 5 - 1) // 4 + 1
    return lat_t * (CROP_OUT // 32) ** 2 <= ONE_CHUNK_TOKENS


def select(dets, n, scale):
    """Tracks worth a close-up -> jobs: {f0, m, cx, cy, side, w, size_out}, at
    most MAX_TRACKS, largest faces first. `scale` = output px per source px."""
    jobs = []
    for tr in link(dets, n):
        if len(jobs) >= MAX_TRACKS:
            break
        w = weights(tr, scale, fade_in=tr["first"] > 0, fade_out=tr["last"] < n - 1)
        if float(w.max()) <= 0.0:
            continue
        nz = torch.nonzero(w > 0).view(-1)
        a, b = int(nz[0]), int(nz[-1]) + 1
        jobs.append({"f0": tr["first"] + a, "m": b - a, "cx": tr["cx"][a:b],
                     "cy": tr["cy"][a:b], "side": tr["size"][a:b] * CONTEXT, "w": w[a:b],
                     "size_out": float(tr["size"][a:b].mean() * scale)})
    return jobs


def crops_from_batches(jobs, batches):
    """Each job's crop clip, cut from source batches read in order. -> list of
    [m, CROP_IN, CROP_IN, 3] (uint8 on the CPU, to keep long tracks small)."""
    out = [torch.empty((j["m"], CROP_IN, CROP_IN, 3), dtype=torch.uint8) for j in jobs]
    at = 0
    for batch in batches:
        k = int(batch.shape[0])
        for j, dst in zip(jobs, out):
            a, b = max(at, j["f0"]), min(at + k, j["f0"] + j["m"])
            if a >= b:
                continue
            i0, i1 = a - j["f0"], b - j["f0"]
            c = crop(batch[a - at:b - at], j["cx"][i0:i1], j["cy"][i0:i1], j["side"][i0:i1])
            dst[i0:i1] = (c * 255.0).round().clamp(0, 255).to(torch.uint8)
        at += k
    return out


def render(jobs, crops, redraw_fn, fps):
    """Run the close-up re-draw for every job; stores job["out"] ([m, S, S, 3] uint8)."""
    for i, (j, c) in enumerate(zip(jobs, crops)):
        t0 = j["f0"] / float(fps)
        out = redraw_fn(c.float().div_(255.0), t0, t0 + j["m"] / float(fps))[: j["m"]]
        # In slices: a close-up that spans a whole long film is tens of GB as floats, and
        # converting it in one go allocated a second full copy (2946 f at 896 px: 28 GB, OOM).
        if out.dtype == torch.uint8:     # the upsampler's close-up is already 8-bit
            q = out
        else:
            q = torch.empty(tuple(out.shape), dtype=torch.uint8)
            for s in range(0, int(out.shape[0]), 64):
                q[s:s + 64] = (out[s:s + 64].float().clamp(0, 1) * 255.0).round().to(torch.uint8)
        del out
        j["out"] = q
        print(f"[{TAG}] small face {i + 1}: frames {j['f0']}-{j['f0'] + j['m'] - 1}, "
              f"{j['size_out']:.0f} px tall in the output, re-drawn as a "
              f"{int(j['out'].shape[1])} px close-up", flush=True)


def paste_jobs(img, f_off, jobs, sx, sy):
    """Blend every rendered job's frames that fall in img (frames f_off ..) in place."""
    k = int(img.shape[0])
    for j in jobs:
        if "out" not in j:
            continue
        a, b = max(f_off, j["f0"]), min(f_off + k, j["f0"] + j["m"])
        if a >= b:
            continue
        i0, i1 = a - j["f0"], b - j["f0"]
        # the 8-bit close-up goes in as is: as floats a whole-film close-up is 28 GB
        paste(img[a - f_off:b - f_off], j["out"][i0:i1], j["cx"][i0:i1],
              j["cy"][i0:i1], j["side"][i0:i1], j["w"][i0:i1], sx, sy)
    return img


def note(jobs):
    done = sum(1 for j in jobs if "out" in j)
    return f", {done} small face(s) re-drawn as close-ups" if done else ", no small faces"


def apply(img, images, redraw_fn, fps=24.0):
    """Re-draw the small faces of `images` (the source, [n, h, w, 3]) as close-ups
    and blend them into `img` (the upsampled clip, [n, Ho, Wo, 3]) in place.

    redraw_fn(crops [m, CROP_IN, CROP_IN, 3], t0_s, t1_s) -> [m, S, S, 3] (float 0..1 or uint8): the
    upsampler run on the crop clip with the soundtrack from t0_s to t1_s, in ONE
    chunk when it fits (see one_chunk): a chunk seam inside a close-up morphs the face.
    -> (img, note)"""
    if not FACE_PASS:
        return img, ""
    n = min(int(img.shape[0]), int(images.shape[0]))
    sh, sw = int(images.shape[1]), int(images.shape[2])
    sx, sy = int(img.shape[2]) / float(sw), int(img.shape[1]) / float(sh)
    dets = detect(images[:n])
    if dets is None:
        return img, ""
    jobs = select(dets, n, (sx + sy) / 2.0)
    if not jobs:
        return img, note(jobs)
    crops = crops_from_batches(jobs, (images[s:s + 64] for s in range(0, n, 64)))
    render(jobs, crops, redraw_fn, fps)
    del crops
    paste_jobs(img, 0, jobs, sx, sy)
    return img, note(jobs)
