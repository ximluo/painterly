"""Depth (Depth Anything V2 on MPS), saliency (BiRefNet via rembg), and the
derived detail/bucket maps that drive stroke ordering. Model outputs are
cached as .npy so reruns skip inference entirely.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import Config

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SALIENCY_MODEL = "birefnet-general-lite"


@dataclass
class Maps:
    nearness: np.ndarray   # float32 [0,1], 1 = nearest (DA V2 is inverse depth)
    saliency: np.ndarray   # float32 [0,1] soft subject matte
    detail: np.ndarray     # float32 [0,1] how much fine detail a pixel earns
    buckets: np.ndarray    # uint8 paint groups: 0 = farthest, painted first;
                           # the salient subject is promoted to the last group
                           # even when something else (a foreground ledge) is
                           # nearer — a painter saves the subject for last
    faces: np.ndarray      # float32 [0,1] feathered face regions (human or
                           # cat); earns extra-fine strokes
    eyes: np.ndarray       # float32 [0,1] feathered eye regions (YuNet
                           # landmarks; synthesized for cats) — the most
                           # refined spots in the whole painting
    face_ids: np.ndarray   # uint8: 0 = no face, k = face k's ellipse; a
                           # paint stroke may only enter its OWN face


def compute_maps(img: np.ndarray, cfg: Config) -> Maps:
    """img: uint8 RGB HxWx3 at working resolution."""
    h, w = img.shape[:2]
    nearness = _cached(cfg, "nearness", lambda: _depth(img, cfg))
    saliency = _cached(cfg, "saliency", lambda: _saliency(img))

    faces, eyes, face_ids = _detect_faces(img, saliency)
    detail = np.clip(cfg.saliency_weight * saliency
                     + cfg.nearness_weight * nearness, 0, 1).astype(np.float32)
    detail = np.maximum(detail, 0.9 * faces)
    detail = np.maximum(detail, eyes)
    # Equal-population quantile buckets — robust to skewed depth histograms —
    # then merge adjacent buckets whose depths barely differ: on a shallow
    # background the quantile lines are arbitrary and would paint as seams.
    qs = np.quantile(nearness, np.linspace(0, 1, cfg.depth_buckets + 1)[1:-1])
    buckets = np.digitize(nearness, qs).astype(np.uint8)
    medians = [float(np.median(nearness[buckets == b])) if (buckets == b).any()
               else np.nan for b in range(cfg.depth_buckets)]
    relabel = [0]
    for b in range(1, cfg.depth_buckets):
        distinct = (np.isfinite(medians[b]) and np.isfinite(medians[b - 1])
                    and medians[b] - medians[b - 1] >= cfg.bucket_merge)
        relabel.append(relabel[-1] + int(distinct))
    buckets = np.array(relabel, np.uint8)[buckets]
    # Close small holes in the matte (a chest pixel left outside the subject
    # would get its strokes scheduled in the LATE background ranks and smear
    # the refined figure — the body version of the v0.13 face bug).
    subj = (saliency > 0.5).astype(np.uint8)
    subj = cv2.morphologyEx(subj, cv2.MORPH_CLOSE,
                            np.ones((15, 15), np.uint8))
    holes = subj.copy()
    ffmask = np.zeros((subj.shape[0] + 2, subj.shape[1] + 2), np.uint8)
    cv2.floodFill(holes, ffmask, (0, 0), 1)
    subj[holes == 0] = 1  # anything flood-fill couldn't reach = interior hole
    buckets[subj > 0] = buckets.max() + 1
    return Maps(nearness, saliency, detail, buckets, faces, eyes, face_ids)


def neutral_maps(h: int, w: int) -> Maps:
    """Classic Hertzmann mode: no depth, no saliency, full detail everywhere."""
    ones = np.ones((h, w), np.float32)
    zeros = np.zeros((h, w), np.float32)
    return Maps(nearness=ones * 0.5, saliency=ones, detail=ones,
                buckets=np.zeros((h, w), np.uint8), faces=zeros,
                eyes=zeros.copy(), face_ids=np.zeros((h, w), np.uint8))


ASSETS = Path(__file__).resolve().parents[2] / "assets"


def _detect_faces(img: np.ndarray,
                  saliency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feathered masks of detected faces and their eyes, restricted to the
    salient subject. YuNet (tiny ONNX, vendored — OpenCV 5 wheels stopped
    bundling detector data) handles tilted and profile human faces and
    provides eye landmarks; a Haar cascade covers cats (eyes synthesized at
    the standard proportions of the face box)."""
    h, w = img.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    eye_pts: list[tuple[float, float, float]] = []  # (x, y, face width)

    det = cv2.FaceDetectorYN.create(str(ASSETS / "face_yunet.onnx"), "",
                                    (w, h), score_threshold=0.6)
    _, found = det.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if found is not None:
        for f in found:
            boxes.append(tuple(int(v) for v in f[:4]))
            e0 = (float(f[4]), float(f[5]))
            e1 = (float(f[6]), float(f[7]))
            fw_ = float(f[2])
            eye_pts.append((*e0, fw_))
            # Profile view: YuNet's two eye landmarks nearly coincide — the
            # second would drop dark paint on the cheek. Keep one eye.
            if np.hypot(e1[0] - e0[0], e1[1] - e0[1]) > 0.3 * fw_:
                eye_pts.append((*e1, fw_))

    cat = cv2.CascadeClassifier(
        str(ASSETS / "haarcascade_frontalcatface_extended.xml"))
    if not cat.empty():
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for x, y, bw, bh in cat.detectMultiScale(gray, scaleFactor=1.05,
                                                 minNeighbors=3,
                                                 minSize=(h // 16, h // 16)):
            boxes.append((int(x), int(y), int(bw), int(bh)))
            eye_pts.append((x + 0.32 * bw, y + 0.44 * bh, float(bw)))
            eye_pts.append((x + 0.68 * bw, y + 0.44 * bh, float(bw)))

    faces = np.zeros((h, w), np.float32)
    eyes = np.zeros((h, w), np.float32)
    face_ids = np.zeros((h, w), np.uint8)
    kept: list[tuple[int, int, int, int]] = []
    for x, y, bw, bh in boxes:
        cx = int(np.clip(x + bw // 2, 0, w - 1))
        cy = int(np.clip(y + bh // 2, 0, h - 1))
        if saliency[cy, cx] < 0.4:
            continue  # face-shaped noise in the background
        kept.append((x, y, bw, bh))
        cv2.ellipse(faces, (cx, cy), (int(bw * 0.6), int(bh * 0.6)),
                    0, 0, 360, 1.0, -1)
        cv2.ellipse(face_ids, (cx, cy), (int(bw * 0.6), int(bh * 0.6)),
                    0, 0, 360, len(kept), -1)
    for ex, ey, fw in eye_pts:
        if not (0 <= ex < w and 0 <= ey < h):
            continue
        if any(x - bw * 0.3 <= ex <= x + 1.3 * bw and y - bh * 0.3 <= ey <= y + 1.3 * bh
               for x, y, bw, bh in kept):
            cv2.circle(eyes, (int(ex), int(ey)), max(3, int(0.13 * fw)),
                       1.0, -1)
    if faces.any():
        faces = np.clip(cv2.GaussianBlur(faces, (0, 0), 15) * 1.4, 0, 1)
    if eyes.any():
        eyes = np.clip(cv2.GaussianBlur(eyes, (0, 0), 5) * 1.6, 0, 1)
    return faces, eyes, face_ids


def _cached(cfg: Config, name: str, compute) -> np.ndarray:
    import hashlib

    tag = hashlib.md5(str(cfg.input_path.resolve()).encode()).hexdigest()[:8]
    path = cfg.cache_dir / f"{cfg.input_path.stem}.{tag}.{cfg.size}.{name}.npy"
    if path.exists():
        return np.load(path)
    arr = compute()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def _depth(img: np.ndarray, cfg: Config) -> np.ndarray:
    from transformers import pipeline  # deferred: torch import is slow

    pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL, device=cfg.device)
    pred = pipe(Image.fromarray(img))["predicted_depth"].squeeze().float().cpu().numpy()
    pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(pred.min()), float(pred.max())
    return ((pred - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def _saliency(img: np.ndarray) -> np.ndarray:
    from rembg import new_session, remove

    session = new_session(SALIENCY_MODEL, providers=["CPUExecutionProvider"])
    mask = remove(Image.fromarray(img), session=session, only_mask=True)
    sal = np.asarray(mask, np.float32) / 255.0
    return cv2.GaussianBlur(sal, (0, 0), sigmaX=4)


def dump_debug(maps: Maps, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("nearness", "saliency", "detail"):
        arr = getattr(maps, name)
        gray = (arr * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{name}.png"),
                    cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS))
    n = int(maps.buckets.max()) + 1
    gray = (maps.buckets.astype(np.float32) / max(n - 1, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "buckets.png"),
                cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS))
