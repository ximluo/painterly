"""Build-up timelapse: ease-in stroke schedule, frames streamed straight to
x264 — never buffered (450 frames of 1080p would be ~2.8 GB in RAM).

The sketch also lives on a separate OVERLAY layer during playback: base
coats visibly go down UNDER the drawing (the Paints-UNDO / digital-painting
lineart-above-flats workflow), and the drawing dissolves as refinement
progresses. The overlay never touches the final canvas.
"""
import dataclasses

import cv2
import numpy as np

from .config import Config
from .render import (blank_canvas, draw_stroke, face_block_masks,
                     soft_bucket_masks, to_uint8)
from .strokes import SKETCH


PAINT_PHASE = 2  # index into phase_fractions; falls heir to empty phases' frames


def stroke_schedule(strokes: list, n_frames: int, cfg: Config) -> list[int]:
    """Cumulative stroke count per frame, allotting screen time per phase
    (wash/sketch/paint/highlight fractions from config). Strokes must
    already be phase-sorted. Wash, sketch and highlights play linearly;
    the paint phase keeps the ease-in power curve (a few big background
    strokes early, fine detail raining in late)."""
    counts = np.bincount([s.phase for s in strokes], minlength=4)
    frames = [int(round(f * n_frames)) for f in cfg.phase_fractions]
    for i, c in enumerate(counts):
        if c == 0 and i != PAINT_PHASE:
            frames[PAINT_PHASE] += frames[i]
            frames[i] = 0
    frames[PAINT_PHASE] += n_frames - sum(frames)

    cum, base = [], 0
    for i, (count, nf) in enumerate(zip(counts, frames)):
        if nf == 0:
            base += int(count)
            continue
        t = np.arange(1, nf + 1) / nf
        prog = t ** cfg.ease_power if i == PAINT_PHASE else t
        cum.extend((base + np.round(count * prog).astype(int)).tolist())
        base += int(count)
    cum[-1] = len(strokes)
    return cum


def write_timelapse(strokes: list, h: int, w: int, cfg: Config,
                    textured: bool = True,
                    buckets: np.ndarray | None = None,
                    ground: np.ndarray | None = None,
                    face_ids: np.ndarray | None = None) -> np.ndarray:
    """Render ordered strokes to out_dir/timelapse.mp4; returns the final
    supersampled canvas (the painting). Sketch strokes never touch the
    canvas: they live on an overlay composited above each frame, and wrong
    lines get their own temp layer that vanishes completely on the undo
    marker (real ctrl-Z). The overlay does NOT fade on a timer — it
    disappears exactly where paint physically covers it (a per-pixel cover
    map accumulates every paint stroke's alpha)."""
    import imageio.v2 as imageio

    ss = cfg.supersample
    # x264 yuv420p needs even dimensions.
    fw, fh = w - w % 2, h - h % 2
    canvas = blank_canvas(h * ss, w * ss, ground)
    soft = soft_bucket_masks(buckets, h * ss, w * ss, 3.0 * ss)
    fblocks = face_block_masks(face_ids, h * ss, w * ss)
    n_build = int(round(cfg.fps * cfg.build_seconds))
    schedule = stroke_schedule(strokes, n_build, cfg)

    overlay3 = np.zeros_like(canvas)   # sketch lines, white-on-black
    temps: dict[int, np.ndarray] = {}  # live wrong-line layers by gid
    cover = np.zeros((h * ss, w * ss), np.float32)
    white = np.full(3, 255.0, np.float32)
    gray = np.array(cfg.sketch_color, np.float32)
    omask = None
    overlay_done = False  # once fully covered it can never come back

    writer = imageio.get_writer(
        str(cfg.out_dir / "timelapse.mp4"), fps=cfg.fps, codec="libx264",
        quality=8, pixelformat="yuv420p", macro_block_size=1,
    )
    done = 0
    try:
        for i, target in enumerate(schedule):
            dirty = False
            for stroke in strokes[done:target]:
                if stroke.phase == SKETCH:
                    dirty = True
                    if stroke.undo < 0:
                        temps.pop(-stroke.undo, None)  # ctrl-Z: gone entirely
                    elif stroke.undo > 0:
                        layer = temps.setdefault(stroke.undo,
                                                 np.zeros_like(overlay3))
                        draw_stroke(layer, dataclasses.replace(
                            stroke, color=white, blend=0.0), scale=ss)
                    else:
                        draw_stroke(overlay3, dataclasses.replace(
                            stroke, color=white, blend=0.0), scale=ss)
                else:
                    draw_stroke(canvas, stroke, scale=ss, textured=textured,
                                soft_masks=soft, face_blocks=fblocks,
                                cover=cover)
            done = target
            frame = to_uint8(canvas, (fw, fh)).astype(np.float32)

            if not overlay_done:
                if dirty or temps or omask is None:
                    total = overlay3
                    for layer in temps.values():
                        total = np.maximum(total, layer)
                    omask = (to_uint8(total, (fw, fh))[..., 0]
                             .astype(np.float32) / 255.0)
                coverf = cv2.resize(cover, (fw, fh),
                                    interpolation=cv2.INTER_AREA)
                wmap = omask * (1.0 - coverf) * 0.9
                if wmap.max() < 0.01 and done > 0 and not temps:
                    overlay_done = omask.max() > 0  # buried for good
                else:
                    m = wmap[..., None]
                    frame = frame * (1.0 - m) + gray * m
            writer.append_data(frame.astype(np.uint8))
        final = to_uint8(canvas, (fw, fh))
        for _ in range(int(round(cfg.fps * cfg.hold_seconds))):
            writer.append_data(final)
    finally:
        writer.close()
    return canvas
