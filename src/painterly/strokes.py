"""Hertzmann '98 curved-brush stroke engine, modernized:

- strokes follow an Edge Tangent Flow field (Kang et al. 2007, as used by
  Im2Oil) instead of raw per-pixel gradients — neighboring strokes come out
  locally parallel, which is what makes marks read as brushwork
- each grown spine is collapsed to a single quadratic Bezier
  (LearningToPaint-style, 3 control points) — no residual wobble
- detail gate: fine brush layers only seed where the detail map is high
- silhouette clip: strokes never grow across a depth discontinuity, which is
  what makes depth-reordered playback artifact-free
"""
import colorsys
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .maps import Maps
from .render import (HIGHLIGHT, PAINT, PAPER_COLOR, SKETCH, WASH, draw_stroke,
                     face_block_masks)

__all__ = ["WASH", "SKETCH", "PAINT", "HIGHLIGHT", "Stroke", "generate_strokes"]


@dataclass
class Stroke:
    points: np.ndarray  # (n, 2) float32, xy at base resolution
    radius: float
    color: np.ndarray   # (3,) float32 RGB 0-255
    alpha: float
    bucket: int         # paint group; 0 for wash/sketch = exempt from masking
    layer: int
    seq: int
    seed_near: float    # nearness at the seed (kept for debugging/analysis)
    phase: int = PAINT
    blend: float = 0.0  # wet mixing: fraction of the underlying canvas color
                        # picked up by the brush (paint phases only)
    firmness: float = 1.0  # 1 = fully loaded big brush, solid swath with
                           # almost no taper; 0 = light expressive stroke
    in_face: bool = False  # seeded inside a detected face region
    in_eye: bool = False   # seeded inside an eye region (painted first
                           # within the face, refined hardest at the end)
    face_id: int = 0       # which face's ellipse the seed sits in (0 = none)
    undo: int = 0          # sketch ctrl-Z: +gid draws onto a temp layer,
                           # -gid removes that layer completely (no residue)


UNPAINTED_ERROR = 300.0  # forces the coarsest layer to cover the whole canvas


def _face_centers(face_ids: np.ndarray) -> dict[int, np.ndarray]:
    return {int(k): np.argwhere(face_ids == k).mean(axis=0)[::-1]
            for k in np.unique(face_ids) if k > 0}


def _face_id_at(maps: Maps, sx: int, sy: int,
                centers: dict[int, np.ndarray]) -> int:
    """The id at the seed, or — for in-face strokes seeded in the feathered
    ring outside the id ellipse (hairline, jaw) — the NEAREST face's id, so
    the protection rule never half-blocks a face's own edge strokes (the
    v0.12 slop bug)."""
    fid = int(maps.face_ids[sy, sx])
    if fid or not centers or maps.faces[sy, sx] <= 0.3:
        return fid
    return min(centers, key=lambda k: (centers[k][0] - sx) ** 2
               + (centers[k][1] - sy) ** 2)


def _etf(gray: np.ndarray, sigma: float):
    """Edge Tangent Flow, cheap double-angle version: orientation is
    pi-periodic so it's blurred in (cos2t, sin2t) space, magnitude-weighted
    so weak/noisy pixels adopt the direction of nearby strong edges.
    Returns (tx, ty, strength)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx) + np.pi / 2
    c = mag * np.cos(2 * theta)
    s = mag * np.sin(2 * theta)
    # Half the brush radius: coarse layers get strongly aligned fields,
    # fine layers keep enough locality to trace small features (eyes!).
    sig = float(np.clip(0.5 * sigma, 1.5, 12.0))
    for _ in range(3):
        c = cv2.GaussianBlur(c, (0, 0), sigmaX=sig)
        s = cv2.GaussianBlur(s, (0, 0), sigmaX=sig)
    half = 0.5 * np.arctan2(s, c)
    return (np.cos(half).astype(np.float32), np.sin(half).astype(np.float32),
            np.hypot(c, s).astype(np.float32))


def _bezier(points: np.ndarray, n: int = 10) -> np.ndarray:
    """Collapse a grown spine to one quadratic Bezier (endpoints + the
    midpoint pulled through the half-arc-length point). Three control points
    is what SOTA painters use — it structurally can't wobble."""
    if len(points) < 3:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    m = np.array([np.interp(cum[-1] / 2, cum, points[:, 0]),
                  np.interp(cum[-1] / 2, cum, points[:, 1])], np.float32)
    p0, p2 = points[0], points[-1]
    p1 = 2 * m - 0.5 * (p0 + p2)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.float32)


def generate_strokes(src: np.ndarray, maps: Maps, cfg: Config,
                     rng: np.random.Generator, phases: bool = True,
                     ground: np.ndarray | None = None) -> list[Stroke]:
    """src: float32 RGB HxWx3 in 0-255 at base resolution. With phases=True
    (the artist pipeline) a full-canvas wash precedes the paint layers and a
    highlight pass follows them; --classic sets phases=False."""
    h, w = src.shape[:2]
    scale = max(h, w) / 1080
    base = ground if ground is not None else np.array(PAPER_COLOR, np.float32)
    canvas = np.full_like(src, 0.0) + base
    painted = np.zeros((h, w), bool)
    top = int(maps.buckets.max())
    fblocks = face_block_masks(maps.face_ids, h, w)
    centers = _face_centers(maps.face_ids)
    strokes: list[Stroke] = []

    if phases:
        _wash_pass(src, canvas, maps, cfg, rng, scale, strokes, base)

    for layer, base_radius in enumerate(cfg.radii):
        radius = max(1.0, base_radius * scale)
        ref = cv2.GaussianBlur(src, (0, 0), sigmaX=0.5 * radius)
        gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
        tx, ty, tstr = _etf(gray, sigma=radius)

        err = np.linalg.norm(canvas - ref, axis=2)
        err[~painted] = UNPAINTED_ERROR
        step = int(round(radius))
        area_err = cv2.boxFilter(err, -1, (step, step))

        seeds = _failing_cells(err, area_err, step, cfg.error_threshold)
        if len(seeds) > cfg.max_strokes_per_layer:
            seeds = seeds[: cfg.max_strokes_per_layer]  # already sorted by error
        rng.shuffle(seeds)

        gate = cfg.detail_gates[layer] if layer < len(cfg.detail_gates) else 0.0
        for sx, sy in seeds:
            detail = float(maps.detail[sy, sx])
            if detail < gate:
                continue
            if (layer >= 3 and maps.face_ids[sy, sx] > 0
                    and rng.random() < 0.45):
                # Full v0.13-style face layering, moderated: the finest
                # layers are thinned so the face never over-renders; the
                # face pass and the sharp eye pass finish it.
                continue
            # Per-stroke variety: width jitter, and length biased shorter in
            # tight (high-detail) areas — dabs on the subject, lazier strokes
            # in the background.
            jradius = radius * rng.uniform(*cfg.radius_jitter)
            high = max(cfg.min_stroke_points + 1,
                       round(cfg.max_stroke_points * (1.0 - 0.45 * detail)))
            rot = rng.normal(0.0, np.radians(cfg.direction_jitter))
            if rng.random() < cfg.cross_rate:
                rot += np.pi / 2  # the occasional cross-stroke
            points = _grow_stroke(sx, sy, jradius, ref, canvas, painted,
                                  tx, ty, tstr, maps.nearness, cfg,
                                  rand_angle=rng.uniform(0, 2 * np.pi),
                                  max_points=int(rng.integers(
                                      cfg.min_stroke_points, high + 1)),
                                  dir_rot=rot)
            sat = cfg.sat_scale[layer] if layer < len(cfg.sat_scale) else 1.0
            color = _jittered_color(ref[sy, sx], detail, cfg, rng,
                                    sat_scale=sat if phases else 1.0)
            alpha = cfg.min_alpha + (1.0 - cfg.min_alpha) * detail
            if layer <= 1:
                alpha = max(alpha, 0.92)  # opaque block-in over the wash
            stroke = Stroke(
                points=_bezier(points),
                radius=jradius,
                color=color,
                alpha=alpha,
                bucket=int(maps.buckets[sy, sx]),
                layer=layer,
                seq=len(strokes),
                seed_near=float(maps.nearness[sy, sx]),
                blend=cfg.blend if phases else 0.0,
                firmness=max(0.4, 1.0 - 0.15 * layer),
                in_face=bool(maps.faces[sy, sx] > 0.3),
                in_eye=bool(maps.eyes[sy, sx] > 0.3),
                face_id=_face_id_at(maps, sx, sy, centers),
            )
            strokes.append(stroke)
            draw_stroke(canvas, stroke, painted=painted, buckets=maps.buckets, top=top, face_blocks=fblocks, fast=True)

    if phases:
        _face_pass(src, canvas, painted, maps, cfg, rng, scale, strokes)
        _eye_pass(src, canvas, painted, maps, cfg, rng, scale, strokes)
        _highlight_pass(src, maps, cfg, rng, scale, strokes)
    return strokes


def _eye_pass(src: np.ndarray, canvas: np.ndarray, painted: np.ndarray,
              maps: Maps, cfg: Config, rng: np.random.Generator,
              scale: float, strokes: list[Stroke]) -> None:
    """The finest strokes in the painting, restricted to the eyes — a
    portrait lives or dies by them. Lands after the face pass."""
    if not maps.eyes.any():
        return
    top = int(maps.buckets.max())
    centers = _face_centers(maps.face_ids)
    radius = max(1.0, cfg.eye_radius * scale)
    ref = cv2.GaussianBlur(src, (0, 0), sigmaX=0.8)
    gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    tx, ty, tstr = _etf(gray, sigma=radius)
    ys, xs = np.where(maps.eyes >= 0.3)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    err = np.linalg.norm(canvas - ref, axis=2)
    err[maps.eyes < 0.3] = 0.0
    step = int(round(radius))
    area_err = cv2.boxFilter(err, -1, (step, step))
    seeds = _failing_cells(err[y0:y1, x0:x1], area_err[y0:y1, x0:x1], step,
                           0.4 * cfg.error_threshold)
    seeds = [(sx + x0, sy + y0) for sx, sy in seeds][:1500]
    rng.shuffle(seeds)
    for sx, sy in seeds:
        points = _grow_stroke(sx, sy, radius, ref, canvas, painted,
                              tx, ty, tstr, maps.nearness, cfg,
                              rand_angle=rng.uniform(0, 2 * np.pi),
                              max_points=6)
        stroke = Stroke(points=_bezier(points), radius=radius,
                        color=ref[sy, sx].astype(np.float32).copy(), alpha=1.0,
                        bucket=int(maps.buckets[sy, sx]),
                        layer=len(cfg.radii) + 1, seq=len(strokes),
                        seed_near=float(maps.nearness[sy, sx]),
                        blend=0.3 * cfg.blend, firmness=0.45,
                        in_face=True, in_eye=True,
                        face_id=_face_id_at(maps, sx, sy, centers))
        strokes.append(stroke)
        draw_stroke(canvas, stroke, painted=painted, buckets=maps.buckets,
                    top=top, fast=True)


def _face_pass(src: np.ndarray, canvas: np.ndarray, painted: np.ndarray,
               maps: Maps, cfg: Config, rng: np.random.Generator,
               scale: float, strokes: list[Stroke]) -> None:
    """One extra-fine layer restricted to detected faces — the face ends up
    visibly more resolved than the rest, the way portrait painters work."""
    if not maps.faces.any():
        return
    top = int(maps.buckets.max())
    centers = _face_centers(maps.face_ids)
    radius = max(1.0, cfg.face_radius * scale)
    ref = cv2.GaussianBlur(src, (0, 0), sigmaX=max(0.5 * radius, 0.8))
    gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    tx, ty, tstr = _etf(gray, sigma=radius)
    ys, xs = np.where(maps.faces >= 0.3)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    err = np.linalg.norm(canvas - ref, axis=2)
    err[maps.faces < 0.3] = 0.0
    step = int(round(radius))
    area_err = cv2.boxFilter(err, -1, (step, step))
    # Scan only the face bounding box — a 2px grid over the full image is slow.
    seeds = _failing_cells(err[y0:y1, x0:x1], area_err[y0:y1, x0:x1], step,
                           0.6 * cfg.error_threshold)
    seeds = [(sx + x0, sy + y0) for sx, sy in seeds]
    seeds = seeds[: cfg.max_strokes_per_layer // 3]
    rng.shuffle(seeds)
    for sx, sy in seeds:
        points = _grow_stroke(sx, sy, radius, ref, canvas, painted,
                              tx, ty, tstr, maps.nearness, cfg,
                              rand_angle=rng.uniform(0, 2 * np.pi),
                              dir_rot=rng.normal(0.0, np.radians(
                                  cfg.direction_jitter)))
        stroke = Stroke(points=_bezier(points), radius=radius,
                        color=ref[sy, sx].astype(np.float32).copy(), alpha=1.0,
                        bucket=int(maps.buckets[sy, sx]), layer=len(cfg.radii),
                        seq=len(strokes),
                        seed_near=float(maps.nearness[sy, sx]),
                        blend=0.5 * cfg.blend, firmness=0.5, in_face=True,
                        face_id=_face_id_at(maps, sx, sy, centers))
        strokes.append(stroke)
        draw_stroke(canvas, stroke, painted=painted, buckets=maps.buckets,
                    top=top, fast=True)


def _wash_pass(src: np.ndarray, canvas: np.ndarray, maps: Maps, cfg: Config,
               rng: np.random.Generator, scale: float,
               strokes: list[Stroke], ground: np.ndarray) -> None:
    """Imprimatura: tone the whole canvas with big thin desaturated washes.
    Doesn't mark pixels painted — the block-in covers it, and its color only
    survives as an undertone through translucent later strokes."""
    h, w = src.shape[:2]
    radius = cfg.wash_radius * scale
    ref = cv2.GaussianBlur(src, (0, 0), sigmaX=2 * radius)
    gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    tx, ty, tstr = _etf(gray, sigma=radius)
    painted = np.zeros((h, w), bool)  # local: wash never stops early

    # Dense grid plus explicit border rows/cols so tapered strokes still
    # tone the canvas to its very edges.
    step = max(1, int(round(0.75 * radius)))
    xs = sorted({*range(0, w, step), w - 1})
    ys = sorted({*range(0, h, step), h - 1})
    seeds = [(min(x + rng.integers(0, step), w - 1), min(y + rng.integers(0, step), h - 1))
             for y in ys for x in xs]
    rng.shuffle(seeds)
    for sx, sy in seeds:
        points = _grow_stroke(sx, sy, radius, ref, canvas, painted,
                              tx, ty, tstr, maps.nearness, cfg,
                              rand_angle=rng.uniform(0, 2 * np.pi),
                              max_points=cfg.wash_max_points)
        color = _desaturate(ref[sy, sx], cfg.wash_desat)
        color = color + cfg.wash_tone_pull * (ground - color)
        stroke = Stroke(points=_bezier(points), radius=radius,
                        color=color, alpha=cfg.wash_alpha, bucket=0, layer=0,
                        seq=len(strokes),
                        seed_near=float(maps.nearness[sy, sx]), phase=WASH)
        strokes.append(stroke)
        draw_stroke(canvas, stroke, fast=True)


def _highlight_pass(src: np.ndarray, maps: Maps, cfg: Config,
                    rng: np.random.Generator, scale: float,
                    strokes: list[Stroke]) -> None:
    """Final accents: the thickest, most opaque, smallest strokes, placed
    where the subject is locally brightest — glints, whiskers, rim light.
    (Bright-only: dark accents read as speckle noise, not painting.)"""
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    score = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=8 * scale)
    score[maps.detail < cfg.highlight_detail] = 0
    tx, ty, tstr = _etf(gray, sigma=4 * scale)
    radius = cfg.highlight_radius * scale
    taken = np.zeros(src.shape[:2], bool)
    painted = np.zeros(src.shape[:2], bool)
    order = np.argsort(score, axis=None)[::-1][: cfg.highlight_count * 20]

    n = 0
    for flat in order:
        if n >= cfg.highlight_count or score.flat[flat] < cfg.highlight_min_accent:
            break
        sy, sx = np.unravel_index(flat, score.shape)
        if taken[sy, sx]:
            continue
        cv2.circle(taken.view(np.uint8), (int(sx), int(sy)),
                   int(12 * radius), 1, -1)
        points = _grow_stroke(int(sx), int(sy), radius, src, src, painted,
                              tx, ty, tstr, maps.nearness, cfg,
                              rand_angle=rng.uniform(0, 2 * np.pi),
                              max_points=12)
        strokes.append(Stroke(points=_bezier(points), radius=radius,
                              color=src[sy, sx].copy(), alpha=0.85,
                              bucket=int(maps.buckets[sy, sx]), layer=len(cfg.radii),
                              seq=len(strokes),
                              seed_near=float(maps.nearness[sy, sx]),
                              phase=HIGHLIGHT, firmness=0.5,
                              blend=0.35))  # melt into the wet paint below
        n += 1


def _failing_cells(err: np.ndarray, area_err: np.ndarray, step: int,
                   threshold: float) -> list[tuple[int, int]]:
    """Cells whose mean error exceeds threshold; seed = argmax-error pixel
    in the cell. Returned sorted by cell error, worst first."""
    h, w = err.shape
    out = []
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            cy, cx = min(y0 + step // 2, h - 1), min(x0 + step // 2, w - 1)
            cell_err = area_err[cy, cx]
            if cell_err <= threshold:
                continue
            cell = err[y0:y0 + step, x0:x0 + step]
            iy, ix = np.unravel_index(np.argmax(cell), cell.shape)
            out.append((cell_err, x0 + int(ix), y0 + int(iy)))
    out.sort(reverse=True)
    return [(x, y) for _, x, y in out]


VANISH_EPS = 2.0  # Sobel magnitude below this = flat region, go straight.
                  # High enough that sensor noise in near-black areas (dark
                  # clothing) doesn't curl strokes into tangles.


def _grow_stroke(x0: int, y0: int, radius: float, ref: np.ndarray,
                 canvas: np.ndarray, painted: np.ndarray,
                 tx: np.ndarray, ty: np.ndarray, tstr: np.ndarray,
                 nearness: np.ndarray, cfg: Config, rand_angle: float,
                 max_points: int | None = None,
                 dir_rot: float = 0.0) -> np.ndarray:
    """Hertzmann's makeSplineStroke, walking the ETF tangent field."""
    h, w = ref.shape[:2]
    color = ref[y0, x0]
    seed_near = nearness[y0, x0]
    x, y = float(x0), float(y0)
    points = [(x, y)]
    last_dx, last_dy = 0.0, 0.0
    step_limit = np.radians(cfg.step_turn_limit)
    turn_budget = np.radians(cfg.max_turn)
    total_turn = 0.0

    for i in range(1, max_points or cfg.max_stroke_points):
        ix, iy = int(x), int(y)
        if i > cfg.min_stroke_points and painted[iy, ix] and (
            np.linalg.norm(ref[iy, ix] - canvas[iy, ix])
            < np.linalg.norm(ref[iy, ix] - color)
        ):
            break
        if abs(float(nearness[iy, ix]) - seed_near) > cfg.silhouette_clip:
            break

        if tstr[iy, ix] < VANISH_EPS:
            # Flat region: long straight strokes read as paint, dots don't.
            if i == 1:
                dx, dy = np.cos(rand_angle), np.sin(rand_angle)
            else:
                dx, dy = last_dx, last_dy
        else:
            dx, dy = float(tx[iy, ix]), float(ty[iy, ix])
            if dir_rot:
                cr, sr = np.cos(dir_rot), np.sin(dir_rot)
                dx, dy = dx * cr - dy * sr, dx * sr + dy * cr
            if dx * last_dx + dy * last_dy < 0:
                dx, dy = -dx, -dy
            fc = cfg.curvature_filter
            dx = fc * dx + (1 - fc) * last_dx
            dy = fc * dy + (1 - fc) * last_dy
            norm = (dx * dx + dy * dy) ** 0.5
            if norm < 1e-6:
                break
            dx, dy = dx / norm, dy / norm
            if i > 1:
                # A brush mark bends only so far: clamp the per-step turn and
                # stop when the total turn is spent — dabs and arcs, not worms.
                dtheta = np.arctan2(last_dx * dy - last_dy * dx,
                                    last_dx * dx + last_dy * dy)
                if abs(dtheta) > step_limit:
                    dtheta = step_limit if dtheta > 0 else -step_limit
                    c, s = np.cos(dtheta), np.sin(dtheta)
                    dx = last_dx * c - last_dy * s
                    dy = last_dx * s + last_dy * c
                total_turn += abs(dtheta)
                if total_turn > turn_budget:
                    break

        x, y = x + radius * dx, y + radius * dy
        if not (0 <= x < w and 0 <= y < h):
            break
        points.append((x, y))
        last_dx, last_dy = dx, dy

    return np.array(points, np.float32)


def _jittered_color(color: np.ndarray, detail: float, cfg: Config,
                    rng: np.random.Generator,
                    sat_scale: float = 1.0) -> np.ndarray:
    """Loose (low-detail) strokes drift in hue/value; tight strokes stay
    true. sat_scale < 1 gives the lean desaturated block-in look — later
    full-saturation layers bring the color back."""
    sigma = cfg.jitter * (1.0 - detail)
    if sigma < 0.5 and sat_scale >= 1.0:
        return color.astype(np.float32).copy()
    r, g, b = (float(c) / 255 for c in color)
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    hue = (hue + rng.normal(0, sigma / 720)) % 1.0
    val = np.clip(val + rng.normal(0, sigma / 255), 0, 1)
    return np.array(colorsys.hsv_to_rgb(hue, sat * sat_scale, val),
                    np.float32) * 255


def _desaturate(color: np.ndarray, amount: float) -> np.ndarray:
    gray = float(color[0]) * 0.299 + float(color[1]) * 0.587 + float(color[2]) * 0.114
    return (color + amount * (gray - color)).astype(np.float32)
