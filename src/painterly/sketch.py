"""Underdrawing: line art extracted from the photo (blurred Canny — at
sketch blur scales this gives clean structural contours), vectorized into
polyline chains, and drawn the way a HAND draws (informed by Paints-UNDO
and drawing pedagogy): light construction/envelope lines first, then each
contour restated in 2-3 wobbly overlapping passes that overshoot corners,
with the occasional wrong line erased and redrawn. One perfect vector line
is the machine tell; the build is the human tell. No models.
"""
import cv2
import numpy as np

from .config import Config
from .maps import Maps
from .strokes import SKETCH, Stroke

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def sketch_strokes(img: np.ndarray, maps: Maps, cfg: Config, seq0: int,
                   rng: np.random.Generator,
                   ground: np.ndarray | None = None
                   ) -> tuple[list[Stroke], np.ndarray]:
    """img: uint8 RGB. Returns (strokes, line-art image for --debug)."""
    h, w = img.shape[:2]
    scale = max(h, w) / 1080

    # Three extraction scales: whole image, finer inside the subject, finest
    # around detected faces — the drawing is densest exactly where an artist
    # spends the most pencil time.
    subj_mask = (maps.saliency > 0.5).astype(np.uint8)
    face_mask = (maps.faces >= 0.3).astype(np.uint8)
    lines = _line_art(img, sigma=1.4 * scale)
    fine = _line_art(img, sigma=0.9 * scale, lo=40, hi=110) * subj_mask
    facef = _line_art(img, sigma=0.6 * scale, lo=35, hi=100) * face_mask

    def to_chains(art, min_len):
        art = _drop_specks(art, min_px=int(20 * scale * scale))
        thin = cv2.ximgproc.thinning(art)
        cs = _trace_chains(thin, min_len=min_len)
        return [seg for c in cs for seg in _split(c, cfg.sketch_max_seg * scale)]

    chains = to_chains(lines, int(12 * scale))
    fine_chains = to_chains(fine, int(10 * scale))
    face_chains = to_chains(facef, int(6 * scale))

    outline_chains = _subject_outline(maps.saliency, scale)
    outline = [seg for c in outline_chains
               for seg in _split(c, cfg.sketch_max_seg * scale)]
    # Face chains get their own small budget (a few angular feature hints);
    # everything else competes in the main budget.
    diag = float(np.hypot(h, w))
    face_chains.sort(key=len, reverse=True)
    face_keep, face_len = [], 0.0
    for c in face_chains:
        if face_len > 1.2 * diag:
            break
        face_keep.append(c)
        face_len += float(len(c))
    scored = sorted(fine_chains + chains + face_chains[len(face_keep):],
                    key=lambda c: (-round(_mean_sal(c, maps.saliency) * 4),
                                   -len(c)))
    face_chains = face_keep
    color = np.array(cfg.sketch_color, np.float32)
    em = _Emitter(maps, cfg, scale, seq0, color)

    # Reference style: ONE tier of light-gray straight rough lines — every
    # chain collapsed to long angular segments, restated once or twice,
    # nothing darker than the gray, with occasional undo events.
    for pts in _construction_lines(outline_chains, maps, scale, rng):
        em.emit(pts, alpha=0.35, radius_mult=1.0, firmness=0.3)

    exempt = len(outline) + len(face_chains)
    budget = cfg.sketch_length_budget * diag
    total = 0.0
    pending = []
    for i, chain in enumerate(outline + face_chains + scored):
        pts = _ghost(chain, scale, cfg.sketch_simplify)
        if len(pts) < 2:
            continue
        if i >= exempt:
            length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            if total + length > budget:
                break
            total += length
        if len(pts) >= 3 and rng.random() < cfg.sketch_erase_rate:
            gid = em.next_gid()
            em.emit(_wrongify(pts, scale, rng), alpha=0.45,
                    radius_mult=1.0, firmness=0.3, undo=gid)
            pending.append((int(rng.integers(2, 4)), gid, pts))
            continue
        _restated_passes(em, pts, cfg, scale, rng, lo=0.3,
                         hi=cfg.sketch_alpha)
        pending = _tick_pending(pending, em, cfg, scale, rng)
    for _, gid, pts in pending:
        em.undo(gid)
        _restated_passes(em, pts, cfg, scale, rng, lo=0.3, hi=cfg.sketch_alpha)
    return em.strokes, lines


def _ghost(chain: np.ndarray, scale: float, eps: float = 10.0) -> np.ndarray:
    """Collapse a contour to long straight segments meeting at angles — the
    rough gesture-sketch look, not contour tracing."""
    approx = cv2.approxPolyDP(chain.reshape(-1, 1, 2).astype(np.int32),
                              eps * scale, False)
    return approx.reshape(-1, 2).astype(np.float32)


class _Emitter:
    def __init__(self, maps, cfg, scale, seq0, color):
        self.maps, self.cfg, self.scale = maps, cfg, scale
        self.seq0, self.color = seq0, color
        self.strokes: list[Stroke] = []
        self._gid = 0

    def next_gid(self) -> int:
        self._gid += 1
        return self._gid

    def emit(self, pts: np.ndarray, alpha: float, radius_mult: float,
             firmness: float, undo: int = 0) -> None:
        if len(pts) < 2:
            return
        h, w = self.maps.nearness.shape
        sx = int(np.clip(pts[0, 0], 0, w - 1))
        sy = int(np.clip(pts[0, 1], 0, h - 1))
        self.strokes.append(Stroke(
            points=pts.astype(np.float32),
            radius=self.cfg.sketch_radius * self.scale * radius_mult,
            color=self.color.copy(),
            alpha=alpha, bucket=0, layer=0,
            seq=self.seq0 + len(self.strokes),
            seed_near=float(self.maps.nearness[sy, sx]), phase=SKETCH,
            firmness=firmness, undo=undo))

    def undo(self, gid: int) -> None:
        """Ctrl-Z marker: the tagged line disappears completely."""
        self.strokes.append(Stroke(
            points=np.zeros((2, 2), np.float32), radius=1.0,
            color=self.color.copy(), alpha=0.0, bucket=0, layer=0,
            seq=self.seq0 + len(self.strokes), seed_near=0.5, phase=SKETCH,
            firmness=0.0, undo=-gid))


def _tick_pending(pending, em: _Emitter, cfg, scale, rng):
    """Count down the 'artist hasn't noticed yet' delay on wrong lines."""
    still = []
    for delay, gid, pts in pending:
        if delay <= 1:
            em.undo(gid)
            _restated_passes(em, pts, cfg, scale, rng, lo=0.3, hi=cfg.sketch_alpha)
        else:
            still.append((delay - 1, gid, pts))
    return still


def _restated_passes(em: _Emitter, pts: np.ndarray, cfg: Config, scale: float,
                     rng: np.random.Generator, lo: float = 0.30,
                     hi: float = 0.55, radius_mult: float = 1.0) -> None:
    """A line is built through overlapping imperfect passes, not drawn once:
    light + wobbly first, darker + truer each restatement, final pass
    overshooting the ends the way a confident stroke does."""
    n = int(rng.integers(cfg.sketch_passes[0], cfg.sketch_passes[1] + 1))
    for k in range(n):
        frac = (k + 1) / n
        amp = cfg.sketch_jitter * scale * (1.0 - 0.75 * frac) + 0.3
        pass_pts = _wobble(_trim(pts, rng, 0.15 * (1 - frac) + 0.03), amp, rng)
        if k == n - 1:
            pass_pts = _overshoot(pass_pts, rng.uniform(0.02, 0.05))
        # Mock pen pressure: every pass carries its own weight and darkness,
        # and its own within-stroke taper.
        em.emit(pass_pts,
                alpha=min(cfg.sketch_alpha, (lo + (hi - lo) * frac)
                          * rng.uniform(0.85, 1.1)),
                radius_mult=radius_mult * rng.uniform(0.7, 1.1),
                firmness=rng.uniform(0.22, 0.45))


def _wobble(pts: np.ndarray, amp: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth low-frequency hand wobble along the polyline's normals."""
    if len(pts) < 3:
        return pts + rng.normal(0, amp * 0.5, pts.shape).astype(np.float32)
    noise = rng.normal(0, 1, len(pts))
    ksize = min(9, len(pts) - (1 - len(pts) % 2))  # odd, <= len(pts)
    kernel = cv2.getGaussianKernel(ksize, ksize / 3.5).ravel()
    noise = np.convolve(noise, kernel, mode="same")[: len(pts)]
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    return (pts + amp * noise[:, None] * normals).astype(np.float32)


def _trim(pts: np.ndarray, rng: np.random.Generator, max_frac: float) -> np.ndarray:
    a = int(len(pts) * rng.uniform(0, max_frac))
    b = int(len(pts) * rng.uniform(0, max_frac))
    out = pts[a: len(pts) - b]
    return out if len(out) >= 2 else pts


def _overshoot(pts: np.ndarray, frac: float) -> np.ndarray:
    if len(pts) < 2:
        return pts
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    d0 = pts[0] - pts[1]
    d1 = pts[-1] - pts[-2]
    d0 /= max(np.linalg.norm(d0), 1e-6)
    d1 /= max(np.linalg.norm(d1), 1e-6)
    return np.vstack([pts[0] + d0 * frac * length, pts,
                      pts[-1] + d1 * frac * length]).astype(np.float32)


def _wrongify(pts: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """The line the artist draws first — misplaced and mis-angled."""
    c = pts.mean(axis=0)
    ang = np.radians(rng.uniform(3, 6)) * rng.choice([-1, 1])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]],
                   np.float32)
    off = rng.uniform(-5, 5, 2) * scale
    return ((pts - c) @ rot.T + c + off).astype(np.float32)


def _construction_lines(outline_chains: list[np.ndarray], maps: Maps,
                        scale: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Light envelope lines boxing in the subject, plus face axis lines —
    the guesses an artist makes before committing to contours."""
    def line(a, b):  # densify so wobble can bend it like a hand-drawn line
        t = np.linspace(0, 1, 8, dtype=np.float32)[:, None]
        return _wobble(_overshoot((1 - t) * np.float32(a) + t * np.float32(b),
                                  0.06), 1.5 * scale, rng)

    out: list[np.ndarray] = []
    for c in outline_chains:
        approx = cv2.approxPolyDP(c.reshape(-1, 1, 2).astype(np.int32),
                                  18 * scale, True).reshape(-1, 2).astype(np.float32)
        segs = [(approx[j], approx[(j + 1) % len(approx)])
                for j in range(len(approx))]
        segs.sort(key=lambda s: -float(np.linalg.norm(s[1] - s[0])))
        out.extend(line(a, b) for a, b in segs[:8])
    if maps.faces.any():
        ys, xs = np.where(maps.faces >= 0.3)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx = (x0 + x1) / 2
        eye_y = y0 + 0.45 * (y1 - y0)
        out.append(line([cx, y0], [cx, y1]))
        out.append(line([x0, eye_y], [x1, eye_y]))
    return out


def _line_art(img: np.ndarray, sigma: float, lo: int = 50,
              hi: int = 130) -> np.ndarray:
    """Structural contours as uint8 (255 = line). Heavy pre-blur suppresses
    fur/texture edges so only shape boundaries survive."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.Canny(blurred, lo, hi, L2gradient=True)


def _drop_specks(lines: np.ndarray, min_px: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= max(min_px, 4))
    keep = keep[keep != 0]
    return np.isin(labels, keep).astype(np.uint8) * 255


def _trace_chains(thin: np.ndarray, min_len: int) -> list[np.ndarray]:
    """Walk the 1px skeleton into 8-connected pixel chains, splitting at
    junctions (a branch stops when it hits an already-visited pixel)."""
    on = thin > 0
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    degree = cv2.filter2D(on.astype(np.float32), -1, kernel)
    visited = ~on  # visiting an off pixel is a no-op
    chains = []

    def walk(y: int, x: int) -> None:
        chain = [(x, y)]
        visited[y, x] = True
        while True:
            for dy, dx in NEIGHBORS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < on.shape[0] and 0 <= nx < on.shape[1] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    chain.append((nx, ny))
                    y, x = ny, nx
                    break
            else:
                break
        if len(chain) >= min_len:
            chains.append(np.array(chain, np.float32))

    ends = np.argwhere(on & (degree == 1))
    for y, x in ends:                       # open curves from their endpoints
        if not visited[y, x]:
            walk(int(y), int(x))
    for y, x in np.argwhere(~visited):      # leftover loops
        if not visited[y, x]:
            walk(int(y), int(x))
    return chains


def _split(chain: np.ndarray, max_len: float) -> list[np.ndarray]:
    n = max(1, int(np.ceil(len(chain) / max_len)))  # ~1px per skeleton step
    return [c for c in np.array_split(chain, n) if len(c) >= 2]


def _simplify(chain: np.ndarray) -> np.ndarray:
    approx = cv2.approxPolyDP(chain.reshape(-1, 1, 2).astype(np.int32), 1.5, False)
    return approx.reshape(-1, 2).astype(np.float32)


def _subject_outline(saliency: np.ndarray, scale: float) -> list[np.ndarray]:
    """Closed silhouette contour(s) of the detected subject."""
    mask = (saliency > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = (30 * scale) ** 2
    return [c.reshape(-1, 2).astype(np.float32)
            for c in contours if cv2.contourArea(c) >= min_area]


def _mean_sal(chain: np.ndarray, saliency: np.ndarray) -> float:
    xs = chain[:, 0].astype(int)
    ys = chain[:, 1].astype(int)
    return float(saliency[ys, xs].mean())
