"""Global stroke ordering.

Sketch, wash, subject base coat, one face at a time (largest first, each
finished in its own turn), each body by downward zone, then background
far-to-near. Reordering is safe because draw_stroke's cut-in and per-face
masks stop a later stroke from smearing a finished region.
"""
import numpy as np

from .strokes import PAINT, SKETCH, WASH


def painter_order(strokes: list, h: int, w: int, cells: int,
                  rng: np.random.Generator,
                  n_layers: int = 5, face_flow: str = "one-go") -> list:
    cell_h, cell_w = h / cells, w / cells
    row_dir = 1 if rng.random() < 0.5 else -1
    top = max((s.bucket for s in strokes), default=0)
    paint = [s for s in strokes if s.phase == PAINT]
    anchors, sizes = _figures(paint, h, w)
    n_f = len(anchors)

    # face_id decides face-turn membership, and must stay the same field the
    # protection mask uses. Anything else lets a protected stroke sort into a
    # later rank and paint over the finished face.
    fid_counts: dict[int, int] = {}
    for s in paint:
        if s.face_id:
            fid_counts[s.face_id] = fid_counts.get(s.face_id, 0) + 1
    face_turn = {fid: i for i, fid in
                 enumerate(sorted(fid_counts, key=lambda k: -fid_counts[k]))}
    n_faces = max(len(face_turn), 1)
    FINAL = 99
    # hold back ~20% of the fine face pass so each face gets a visible polish
    # at the very end rather than finishing all at once mid-painting
    deferred = ({s.seq for s in paint
                 if s.face_id and s.layer == n_layers
                 and rng.random() < 0.2}
                if face_flow == "v11" else set())

    def fig(s) -> int:
        p = s.points[0]
        return min(range(n_f),
                   key=lambda i: (p[0] - anchors[i][0]) ** 2
                   + (p[1] - anchors[i][1]) ** 2)

    BG_BASE = 1 + n_faces

    def rank(s) -> int:
        if s.layer == 0:
            if s.bucket == top or s.face_id:
                return 0                          # subject base first
            return BG_BASE                        # bg base waits for faces
        if s.face_id:
            if face_flow == "v11" and (s.layer > n_layers
                                       or s.seq in deferred):
                return FINAL                      # end-of-painting polish
            if face_flow == "together":
                return FINAL if s.layer >= n_layers else 1
            return 1 + face_turn[s.face_id]       # a face's own turn
        if s.bucket == top:
            return 2 + n_faces + fig(s)           # then each figure's body
        return 2 + n_faces + n_f + s.bucket       # then background

    def zone(s, fi: int) -> int:
        """Hair/head surround, then bands running down the body."""
        ax, ay = anchors[fi]
        fs = sizes[fi]
        dy = (float(s.points[0, 1]) - ay) / fs
        dx = abs(float(s.points[0, 0]) - ax) / fs
        if dy < 0.8 and dx < 1.8:
            return 0
        return 1 + int(max(dy, 0.0) / 1.2)

    def serp(s) -> int:
        cx = min(int(s.points[0, 0] / cell_w), cells - 1)
        cy = min(int(s.points[0, 1] / cell_h), cells - 1)
        if row_dir < 0:
            cy = cells - 1 - cy
        return cy * cells + (cx if cy % 2 == 0 else cells - 1 - cx)

    # fine layers walk a serpentine so small brushes finish one patch before
    # moving on; coarse layers (0-2) keep free order
    f_cells = cells * 2

    def sec(s) -> int:
        if s.layer < 3:
            return 0
        cx = min(int(s.points[0, 0] * f_cells / w), f_cells - 1)
        cy = min(int(s.points[0, 1] * f_cells / h), f_cells - 1)
        if row_dir < 0:
            cy = f_cells - 1 - cy
        return cy * f_cells + (cx if cy % 2 == 0 else f_cells - 1 - cx)

    def key(s):
        if s.phase in (SKETCH, WASH):
            return (s.phase, 0, 0, 0, 0, 0.0, s.seq)
        r = rank(s)
        if r == FINAL:             # per-face polish, eye pass last
            return (s.phase, FINAL, face_turn.get(s.face_id, 0), s.layer,
                    sec(s), rng.random(), 0)
        if 1 <= r <= n_faces:      # a face: coarse->fine, eyes lead layers
            return (s.phase, r, s.layer, 0 if s.in_eye else 1, sec(s),
                    rng.random(), 0)
        if n_faces + 1 < r <= n_faces + 1 + n_f:   # a figure's body
            return (s.phase, r, zone(s, r - 2 - n_faces), s.layer, sec(s),
                    rng.random(), 0)
        if r == 0:                 # subject base coat, under the drawing
            return (s.phase, 0, 0, serp(s), s.layer, rng.random(), 0)
        return (s.phase, r, 0, serp(s), s.layer, rng.random(), 0)

    ordered = sorted(strokes, key=key)

    # Weave some elsewhere-strokes into the face turns so the painting doesn't
    # look frozen outside the face. Body blocks are coarse only (layer 1), so
    # the face stays the most resolved region throughout; the background base
    # is held to the last quarter.
    face_lo = next((i for i, s in enumerate(ordered)
                    if s.phase == PAINT and 1 <= rank(s) <= n_faces), None)
    if face_lo is not None:
        face_hi = max(i for i, s in enumerate(ordered)
                      if s.phase == PAINT and 1 <= rank(s) <= n_faces)
        n_seg = face_hi - face_lo + 1
        body_ix = [i for i, s in enumerate(ordered)
                   if i > face_hi and s.phase == PAINT and s.layer == 1
                   and s.bucket == top and rng.random() < 0.35]
        bg_ix = [i for i, s in enumerate(ordered)
                 if i > face_hi and rank(s) == BG_BASE]

        def assign(idxs, lo_frac):
            if not idxs:
                return []
            lo = int(lo_frac * n_seg)
            slots = sorted(rng.integers(lo, n_seg, size=len(idxs)).tolist())
            return list(zip(slots, idxs))  # sorted slots keep group order

        inserts = sorted(assign(body_ix, 0.0) + assign(bg_ix, 0.75),
                         key=lambda t: t[0])
        if inserts:
            face_seg = ordered[face_lo:face_hi + 1]
            woven, bi = [], 0
            for fi, s in enumerate(face_seg):
                while bi < len(inserts) and inserts[bi][0] <= fi:
                    woven.append(ordered[inserts[bi][1]])
                    bi += 1
                woven.append(s)
            woven.extend(ordered[t[1]] for t in inserts[bi:])
            taken = {t[1] for t in inserts}
            ordered = (ordered[:face_lo] + woven
                       + [s for i, s in enumerate(ordered)
                          if i > face_hi and i not in taken])
    return ordered


def _figures(paint: list, h: int, w: int):
    """Face anchors (largest first) and per-face sizes.

    Falls back to the subject centroid when no face was detected.
    """
    diag = float(np.hypot(h, w))
    face_pts = np.array([s.points[0] for s in paint if s.in_face], np.float32)
    if len(face_pts) == 0:
        top = max((s.bucket for s in paint), default=0)
        subj = np.array([s.points[0] for s in paint if s.bucket == top],
                        np.float32)
        c = subj.mean(axis=0) if len(subj) else np.zeros(2, np.float32)
        return [c], [0.25 * diag]
    spread = max(float(face_pts[:, 0].max() - face_pts[:, 0].min()),
                 float(face_pts[:, 1].max() - face_pts[:, 1].min()), 1.0)
    anchors: list[np.ndarray] = []
    for p in face_pts:
        if all(np.hypot(*(p - a)) > 0.35 * spread for a in anchors):
            anchors.append(p.copy())
    if not anchors:
        anchors = [face_pts.mean(axis=0)]
    dists = np.array([[float(np.hypot(*(p - a))) for a in anchors]
                      for p in face_pts])
    assign = dists.argmin(axis=1)
    counts = np.bincount(assign, minlength=len(anchors))
    anchors = [anchors[i] for i in np.argsort(-counts)]  # largest face first
    dists = np.array([[float(np.hypot(*(p - a))) for a in anchors]
                      for p in face_pts])
    assign = dists.argmin(axis=1)
    sizes = []
    for i, a in enumerate(anchors):
        mine = dists[assign == i, i]
        sizes.append(max(float(mine.max()) if len(mine) else 0.0, 0.05 * diag))
    return anchors, sizes
