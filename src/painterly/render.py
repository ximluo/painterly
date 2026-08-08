"""Stroke rasterization, ROI-scoped.

Two paths: a fast flat one (AA polyline, uniform width — used while
GENERATING strokes, where the canvas only feeds error maps), and the final
path: each stroke is a TAPERED TEXTURED RIBBON — a canonical bristle
texture warped piecewise along the Bezier spine (the Stylized Neural
Painting / Paint Transformer mechanism), with width and opacity following a
natural pressure profile — lognormal-ish taper at both ends (Plamondon's
kinematic theory of hand movements), widening through curves (the
two-thirds power law), plus a little deterministic tremor. Bristle streaks
run ALONG the stroke, which is what makes a mark read as a brush stroke
rather than a stamped sausage. Sketch lines keep a soft-disc pencil look.
"""
from collections.abc import Callable, Iterable
from functools import lru_cache

import cv2
import numpy as np

PAPER_COLOR = (245, 242, 235)  # warm off-white canvas

# Playback phases, in the order a painter works: the drawing goes down on
# the bare canvas first; wash and paint then appear UNDER it (the sketch
# lives on the timelapse overlay layer, never on the painting itself).
SKETCH, WASH, PAINT, HIGHLIGHT = 0, 1, 2, 3


def blank_canvas(h: int, w: int,
                 ground: np.ndarray | None = None) -> np.ndarray:
    return np.full((h, w, 3), ground if ground is not None else PAPER_COLOR,
                   np.float32)


def ground_color(src: np.ndarray) -> np.ndarray:
    """Adaptive imprimatura tone: the image's mean color, half-desaturated,
    clamped to a mid value — darker scenes get a darker ground, and gaps in
    coverage read as toned canvas instead of glowing white paper."""
    mean = src.reshape(-1, 3).mean(axis=0).astype(np.float32)
    gray = float(mean @ np.array([0.299, 0.587, 0.114], np.float32))
    c = mean + 0.5 * (gray - mean)
    v = float(c.max())
    return c * (float(np.clip(v, 110.0, 185.0)) / max(v, 1e-3))


def draw_stroke(canvas: np.ndarray, stroke, scale: float = 1.0,
                painted: np.ndarray | None = None,
                textured: bool = True,
                buckets: np.ndarray | None = None,
                top: int = 0,
                soft_masks: dict[int, np.ndarray] | None = None,
                face_blocks: dict[int, np.ndarray] | None = None,
                cover: np.ndarray | None = None,
                fast: bool = False) -> None:
    """buckets (same HxW as canvas) enforces the painter's rule: a stroke may
    slop onto planes that are painted later (they repaint over it) but never
    onto finished ones. With subject-first ordering that means background
    strokes also CUT IN around the already-established subject. Wash and
    sketch phases bypass the rule — they cover the whole canvas by design.
    soft_masks (per-bucket feathered masks from soft_bucket_masks) applies
    the same rule with a soft edge — hard cuts read as seams in the render."""
    h, w = canvas.shape[:2]
    pts = stroke.points * scale
    radius = stroke.radius * scale
    pad = int(np.ceil(radius * 1.4)) + 2
    x0 = max(int(pts[:, 0].min()) - pad, 0)
    y0 = max(int(pts[:, 1].min()) - pad, 0)
    x1 = min(int(np.ceil(pts[:, 0].max())) + pad, w)
    y1 = min(int(np.ceil(pts[:, 1].max())) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return

    local = pts - (x0, y0)
    if fast:
        # Slightly thinner than the real render so error-driven placement
        # over-covers relative to the tapered pressure strokes.
        mask = _flat_mask((y1 - y0, x1 - x0), local, radius * 0.7)
    else:
        mask = _pressure_mask((y1 - y0, x1 - x0), local, radius, stroke,
                              textured)

    a = mask * stroke.alpha
    if (face_blocks is not None and stroke.phase == PAINT
            and stroke.layer > 0):
        # Each face is painted only by its own strokes: once refined,
        # nothing later — not even the neighboring face's turn — may smear
        # it (the layer-0 base coat precedes all faces and is exempt;
        # highlights are a different phase).
        a *= 1.0 - face_blocks.get(stroke.face_id,
                                   face_blocks[0])[y0:y1, x0:x1]
    if stroke.phase >= PAINT:
        if soft_masks is not None:
            a *= soft_masks[stroke.bucket][y0:y1, x0:x1]
        elif buckets is not None:
            broi = buckets[y0:y1, x0:x1]
            allowed = broi >= stroke.bucket
            if stroke.bucket != top:
                allowed &= broi != top  # cut in around the subject
            a *= allowed
    a = a[..., None]
    roi = canvas[y0:y1, x0:x1]
    color = stroke.color
    if stroke.blend > 0.0:
        # Wet mixing: the brush picks up the paint below. Sampling along the
        # spine is as plausible as a full-area mean and much cheaper.
        sx = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        sy = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        color = (1.0 - stroke.blend) * color \
            + stroke.blend * canvas[sy, sx].mean(axis=0)
    roi[:] = a * color + (1.0 - a) * roi
    if cover is not None:
        # A thin wash doesn't erase a pencil line — it only veils it; solid
        # paint buries it.
        ca = a[..., 0] * (0.35 if stroke.phase == WASH else 1.0)
        c = cover[y0:y1, x0:x1]
        c[:] = ca + (1.0 - ca) * c
    if painted is not None:
        painted[y0:y1, x0:x1] |= mask > 0.05


def _flat_mask(shape: tuple[int, int], pts: np.ndarray,
               radius: float) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    ipts = np.round(pts).astype(np.int32)
    if len(ipts) > 1:
        cv2.polylines(mask, [ipts], False, 255,
                      max(1, int(round(2 * radius))), cv2.LINE_AA)
    r = max(1, int(round(radius)))
    cv2.circle(mask, tuple(ipts[0]), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, tuple(ipts[-1]), r, 255, -1, cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


def _pressure_mask(shape: tuple[int, int], pts: np.ndarray, radius: float,
                   stroke, textured: bool) -> np.ndarray:
    f = stroke.firmness
    w_floor, w_span = 0.6 + 0.3 * f, 0.45 - 0.35 * f
    a_floor = 0.55 + 0.35 * f
    if textured and stroke.phase != SKETCH:
        centers, tangents, ts = _resample(pts, spacing=max(1.5, 0.7 * radius))
        p = _pressure(ts, tangents, stroke.seq, stroke.firmness)
        if len(centers) >= 2:
            half = np.maximum(radius * (w_floor + w_span * p), 1.0)
            alphas = a_floor + (1.0 - a_floor) * np.minimum(p, 1.0)
            return _ribbon_mask(shape, centers, tangents, ts, half, alphas)
        # degenerate single-point stroke: fall through to a dab
    centers, tangents, ts = _resample(pts, spacing=max(1.0, 0.4 * radius))
    p = _pressure(ts, tangents, stroke.seq, stroke.firmness)
    mask = np.zeros(shape, np.float32)
    for (cx, cy), pi in zip(centers, p):
        size = max(3, int(round(2 * radius * (w_floor + w_span * pi))))
        _max_stamp(mask, _soft_disc(size) * (a_floor + (1.0 - a_floor) * min(pi, 1.0)),
                   int(cx - size / 2), int(cy - size / 2))
    return np.clip(mask, 0.0, 1.0)


def _ribbon_mask(shape: tuple[int, int], centers: np.ndarray,
                 tangents: np.ndarray, ts: np.ndarray, half: np.ndarray,
                 alphas: np.ndarray) -> np.ndarray:
    """Warp arc-length slices of the canonical bristle texture onto the
    quads of the widening/tapering ribbon (SNP/PaintTransformer mechanism,
    piecewise so streaks follow the curve)."""
    tex = _bristle_texture()
    th, tw = tex.shape
    mask = np.zeros(shape, np.float32)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    for i in range(len(centers) - 1):
        dst = np.float32([
            centers[i] + half[i] * normals[i],
            centers[i + 1] + half[i + 1] * normals[i + 1],
            centers[i + 1] - half[i + 1] * normals[i + 1],
            centers[i] - half[i] * normals[i],
        ])
        qx0 = max(int(dst[:, 0].min()) - 1, 0)
        qy0 = max(int(dst[:, 1].min()) - 1, 0)
        qx1 = min(int(np.ceil(dst[:, 0].max())) + 1, shape[1])
        qy1 = min(int(np.ceil(dst[:, 1].max())) + 1, shape[0])
        if qx1 - qx0 < 1 or qy1 - qy0 < 1:
            continue
        src = np.float32([[ts[i] * (tw - 1), 0], [ts[i + 1] * (tw - 1), 0],
                          [ts[i + 1] * (tw - 1), th - 1], [ts[i] * (tw - 1), th - 1]])
        m = cv2.getPerspectiveTransform(src, dst - np.float32([qx0, qy0]))
        piece = cv2.warpPerspective(tex, m, (qx1 - qx0, qy1 - qy0),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        roi = mask[qy0:qy1, qx0:qx1]
        np.maximum(roi, piece * alphas[i], out=roi)
    return np.clip(mask, 0.0, 1.0)


@lru_cache(maxsize=1)
def _bristle_texture(th: int = 96, tw: int = 384) -> np.ndarray:
    """Canonical grayscale brush texture, bristle streaks along x, soft
    lateral falloff, ragged ends. Deterministic (fixed seed)."""
    r = np.random.default_rng(11)
    rows = r.uniform(0.5, 1.0, th).astype(np.float32)
    tex = cv2.GaussianBlur(np.repeat(rows[:, None], tw, axis=1), (0, 0),
                           sigmaX=7, sigmaY=1.1)
    grain = cv2.GaussianBlur(r.uniform(0, 1, (th, tw)).astype(np.float32),
                             (0, 0), sigmaX=9, sigmaY=2)
    tex = tex * (0.72 + 0.28 * grain)
    lat = np.linspace(-1, 1, th, dtype=np.float32)[:, None]
    tex *= np.clip((1.0 - np.abs(lat)) * 3.5, 0, 1) ** 0.7
    x = np.linspace(0, 1, tw, dtype=np.float32)[None, :]
    ragged = cv2.GaussianBlur(r.uniform(0, 1, (th, tw)).astype(np.float32),
                              (0, 0), sigmaX=4)
    ends = np.clip(x / 0.03, 0, 1) * np.clip((1 - x) / 0.05, 0, 1)
    tex *= np.clip(ends * (0.85 + 0.3 * ragged), 0, 1)
    return np.clip(tex, 0, 1).astype(np.float32)


def _pressure(ts: np.ndarray, tangents: np.ndarray, seq: int,
              firmness: float = 1.0) -> np.ndarray:
    """Per-stamp pressure in ~[0.05, 1.3]: quick landing, longer lift-off,
    heavier through curves, seeded tremor. A firm (fully loaded, large)
    brush barely tapers; a light one breathes."""
    t_in = 0.25 - 0.19 * firmness
    t_out = 0.32 - 0.24 * firmness
    taper = np.clip(ts / t_in, 0, 1) ** 0.6 * np.clip((1 - ts) / t_out, 0, 1) ** 0.6
    ang = np.arctan2(tangents[:, 1], tangents[:, 0])
    dtheta = np.abs(np.diff(ang, prepend=ang[0]))
    dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
    curve = np.clip((1 + 5 * dtheta) ** (1 / 3), 1.0, 1.35)
    tremor = 1 + 0.06 * np.sin(2 * np.pi * (0.37 * np.arange(len(ts)) + 0.618 * seq))
    return np.clip(taper * curve * tremor, 0.05, 1.3)


def _resample(pts: np.ndarray, spacing: float):
    """Even resampling of a polyline: (points, unit tangents, t in [0,1])."""
    if len(pts) < 2:
        return pts[:1], np.array([[1.0, 0.0]], np.float32), np.array([0.5])
    seg = np.diff(pts, axis=0)
    seg_len = np.maximum(np.linalg.norm(seg, axis=1), 1e-6)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < 1e-3:
        return pts[:1], np.array([[1.0, 0.0]], np.float32), np.array([0.5])
    n = max(2, int(total / spacing) + 1)
    t = np.linspace(0.0, total, n)
    idx = np.clip(np.searchsorted(cum, t, "right") - 1, 0, len(seg) - 1)
    frac = (t - cum[idx]) / seg_len[idx]
    centers = pts[idx] + frac[:, None] * seg[idx]
    tangents = seg[idx] / seg_len[idx][:, None]
    return centers, tangents, t / total


@lru_cache(maxsize=256)
def _soft_disc(size: int) -> np.ndarray:
    r = np.linspace(-1, 1, size)
    d = np.sqrt(r[None, :] ** 2 + r[:, None] ** 2)
    return np.clip((1.0 - d) * 2.5, 0, 1).astype(np.float32) ** 0.8


def _max_stamp(mask: np.ndarray, tip: np.ndarray, x0: int, y0: int) -> None:
    th, tw = tip.shape
    bx0, by0 = max(-x0, 0), max(-y0, 0)
    bx1 = min(tw, mask.shape[1] - x0)
    by1 = min(th, mask.shape[0] - y0)
    if bx1 <= bx0 or by1 <= by0:
        return
    dst = mask[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1]
    np.maximum(dst, tip[by0:by1, bx0:bx1], out=dst)


def face_block_masks(face_ids: np.ndarray | None, h: int,
                     w: int) -> dict[int, np.ndarray] | None:
    """Per-face-id feathered 'keep out' masks: blocked[0] covers every face
    (for strokes belonging to none), blocked[k] covers every face EXCEPT
    face k — a stroke may only ever paint inside its own face, so finishing
    one face and then painting its neighbor can never smear the first."""
    if face_ids is None or not face_ids.any():
        return None
    ids = face_ids
    if ids.shape != (h, w):
        ids = cv2.resize(ids, (w, h), interpolation=cv2.INTER_NEAREST)
    union = cv2.GaussianBlur((ids > 0).astype(np.float32), (0, 0), sigmaX=6.0)
    blocked = {0: union}
    for k in (int(v) for v in np.unique(ids) if v > 0):
        own = cv2.GaussianBlur((ids == k).astype(np.float32), (0, 0),
                               sigmaX=6.0)
        blocked[k] = np.clip(union - own, 0.0, 1.0)
    return blocked


def scaled_buckets(buckets: np.ndarray | None, h: int, w: int) -> np.ndarray | None:
    if buckets is None or buckets.shape == (h, w):
        return buckets
    return cv2.resize(buckets, (w, h), interpolation=cv2.INTER_NEAREST)


def soft_bucket_masks(buckets: np.ndarray | None, h: int, w: int,
                      sigma: float) -> dict[int, np.ndarray] | None:
    """Per-bucket feathered painter's-rule masks. Background strokes may
    slop onto later-painted background planes but are cut in around the
    subject (top bucket), which is established before them."""
    if buckets is None:
        return None
    bmap = scaled_buckets(buckets, h, w)
    top = int(bmap.max())
    out = {}
    for b in (int(v) for v in np.unique(bmap)):
        allowed = bmap >= b if b == top else (bmap >= b) & (bmap != top)
        out[b] = cv2.GaussianBlur(allowed.astype(np.float32), (0, 0),
                                  sigmaX=sigma)
    return out


def replay(strokes: Iterable, h: int, w: int, scale: float = 1.0,
           textured: bool = True,
           buckets: np.ndarray | None = None, fast: bool = False,
           ground: np.ndarray | None = None,
           face_ids: np.ndarray | None = None,
           on_stroke: Callable[[int], None] | None = None) -> np.ndarray:
    """Re-rasterize strokes in the given order onto a blank canvas at
    h*scale x w*scale. on_stroke(i) fires after stroke i is drawn."""
    ch, cw = int(round(h * scale)), int(round(w * scale))
    canvas = blank_canvas(ch, cw, ground)
    soft = None if fast else soft_bucket_masks(buckets, ch, cw, 3.0 * scale)
    bmap = scaled_buckets(buckets, ch, cw) if fast else None
    fblocks = face_block_masks(face_ids, ch, cw)
    for i, stroke in enumerate(strokes):
        draw_stroke(canvas, stroke, scale=scale, textured=textured,
                    buckets=bmap, soft_masks=soft, face_blocks=fblocks,
                    fast=fast)
        if on_stroke is not None:
            on_stroke(i)
    return canvas


def to_uint8(canvas: np.ndarray, out_size: tuple[int, int] | None = None) -> np.ndarray:
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    if out_size is not None and (img.shape[1], img.shape[0]) != out_size:
        img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
    return img
