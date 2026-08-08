import argparse
import dataclasses
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import Config
from .maps import compute_maps, dump_debug, neutral_maps
from .order import painter_order
from .render import ground_color, replay, to_uint8
from .sketch import sketch_strokes
from .strokes import SKETCH, WASH, generate_strokes
from .timelapse import write_timelapse


def main() -> None:
    p = argparse.ArgumentParser(prog="painterly",
                                description="Photo -> painting with depth-ordered strokes")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output dir (default: out/<image name>)")
    p.add_argument("--size", type=int, default=1080)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--classic", action="store_true",
                   help="plain Hertzmann '98: no depth, no saliency")
    p.add_argument("--flat", action="store_true",
                   help="disable the textured brush (flat AA strokes)")
    p.add_argument("--debug", action="store_true", help="dump intermediate maps")
    p.add_argument("--face-flow", default="one-go",
                   choices=["v11", "one-go", "together"],
                   help="how faces are sequenced (default: v11)")
    args = p.parse_args()

    cfg = Config(
        input_path=args.input,
        out_dir=args.out or Path("out") / args.input.stem,
        size=args.size, seed=args.seed, device=args.device,
        supersample=args.supersample, video=not args.no_video,
        texture=not args.flat, debug=args.debug, face_flow=args.face_flow,
    )
    run(cfg, classic=args.classic)


def run(cfg: Config, classic: bool = False) -> None:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    t0 = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    img = load_image(cfg.input_path, cfg.size)
    h, w = img.shape[:2]
    rng = np.random.default_rng(cfg.seed)

    if classic:
        maps = neutral_maps(h, w)
    else:
        maps = compute_maps(img, cfg)
        print(f"maps      {time.perf_counter() - t0:6.1f}s")
    if cfg.debug and not classic:
        dump_debug(maps, cfg.debug_dir)

    ground = None if classic else ground_color(img.astype(np.float32))
    strokes = generate_strokes(img.astype(np.float32), maps, cfg, rng,
                               phases=not classic, ground=ground)
    if not classic:
        sketch, line_art = sketch_strokes(img, maps, cfg, seq0=len(strokes),
                                          rng=rng, ground=ground)
        strokes += sketch
        if cfg.debug:
            cfg.debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cfg.debug_dir / "sketch.png"), 255 - line_art)
    per_phase = np.bincount([s.phase for s in strokes], minlength=4)
    print(f"strokes   {time.perf_counter() - t0:6.1f}s   {len(strokes)} total, "
          f"sketch/wash/paint/highlight {per_phase.tolist()}")

    ordered = painter_order(strokes, h, w, cfg.coherence_cells, rng,
                            n_layers=len(cfg.radii),
                            face_flow=cfg.face_flow)
    # The canvas starts as light paper — the sketch goes down first, then
    # the wash visibly tones it under the drawing. `ground` still tints the
    # wash colors themselves.
    if cfg.video:
        canvas = write_timelapse(ordered, h, w, cfg, textured=cfg.texture,
                                 buckets=maps.buckets, ground=None,
                                 face_ids=maps.face_ids)
        print(f"timelapse {time.perf_counter() - t0:6.1f}s   {cfg.out_dir / 'timelapse.mp4'}")
    else:
        # The sketch lives only on the video overlay — the painting itself
        # never receives it.
        canvas = replay([s for s in ordered if s.phase != SKETCH], h, w,
                        scale=cfg.supersample, textured=cfg.texture,
                        buckets=maps.buckets, ground=None,
                        face_ids=maps.face_ids)

    painting = to_uint8(canvas, (w - w % 2, h - h % 2))
    out = cfg.out_dir / "painting.png"
    cv2.imwrite(str(out), cv2.cvtColor(painting, cv2.COLOR_RGB2BGR))
    if cfg.debug:
        dump_stroke_order([s for s in ordered if s.phase != SKETCH],
                          h, w, cfg.debug_dir)
        washes = [s for s in ordered if s.phase == WASH]
        if washes:
            wash_img = to_uint8(replay(washes, h, w))
            cv2.imwrite(str(cfg.debug_dir / "wash.png"),
                        cv2.cvtColor(wash_img, cv2.COLOR_RGB2BGR))
    print(f"done      {time.perf_counter() - t0:6.1f}s   {out}")


def load_image(path: Path, size: int) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = size / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
    return np.asarray(img)


def dump_stroke_order(ordered: list, h: int, w: int, debug_dir: Path) -> None:
    """Strokes colored by playback rank (viridis): should sweep far -> near."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    n = max(len(ordered) - 1, 1)
    ranks = (np.arange(len(ordered)) / n * 255).astype(np.uint8)
    lut = cv2.applyColorMap(ranks.reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
    tinted = [dataclasses.replace(s, color=lut[i].astype(np.float32), alpha=1.0)
              for i, s in enumerate(ordered)]
    canvas = replay(tinted, h, w)
    cv2.imwrite(str(debug_dir / "stroke_order.png"), to_uint8(canvas))  # lut is BGR


if __name__ == "__main__":
    main()
