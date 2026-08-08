from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    input_path: Path
    out_dir: Path
    size: int = 1080                    # long-edge resize of the working image
    seed: int = 0
    device: str = "mps"                 # depth model device: "mps" | "cpu"
    debug: bool = False
    video: bool = True
    texture: bool = True
    supersample: int = 2                # canvas scale; frames downsampled for AA

    blend: float = 0.15                 # wet mixing with the canvas underneath
    face_radius: float = 2.0            # extra-fine brush for detected faces
    eye_radius: float = 1.3             # finest brush of all, eyes only

    # Hertzmann engine
    radii: tuple[int, ...] = (64, 32, 16, 8, 4)   # brush radii at size=1080
    error_threshold: float = 20.0       # area-error trigger, 0-255 color units
    max_stroke_points: int = 12
    min_stroke_points: int = 4
    curvature_filter: float = 0.7
    max_strokes_per_layer: int = 12000
    radius_jitter: tuple[float, float] = (0.65, 1.35)  # per-stroke width variety
    step_turn_limit: float = 28.0       # deg; max direction change per step
    max_turn: float = 60.0              # deg; total turn budget (SOTA painters
                                        # cap ~45-60 deg — beyond that = worms)
    subject_blockin_layers: int = 2     # subject layers painted BEFORE the
                                        # background (figure established first)
    direction_jitter: float = 0.0       # deg; per-stroke rotation off the ETF
                                        # field (0 = the clean v0.13 look)
    cross_rate: float = 0.0             # fraction of near-perpendicular
                                        # cross-strokes (0 = v0.13 look)
    face_flow: str = "one-go"           # v11: one face at a time to ~95%,
                                        #      end-of-painting polish + eyes
                                        # one-go: each face 100% in its turn
                                        # together: all faces interleaved,
                                        #      polish at the end
    face_return_frac: float = 0.35      # slice of the fine face pass held
                                        # back: face hits ~85%% in its turn,
                                        # the return after the background
                                        # brings it to ~95%%

    # Depth / saliency modulation
    depth_buckets: int = 4
    bucket_merge: float = 0.07          # merge buckets whose median nearness is closer
    detail_gates: tuple[float, ...] = (0.0, 0.0, 0.0, 0.45, 0.62)  # per radius layer
    saliency_weight: float = 0.6
    nearness_weight: float = 0.4
    silhouette_clip: float = 0.10       # stop stroke when |nearness - seed| exceeds this
    min_alpha: float = 0.90             # background stroke opacity (lerps to 1.0 with detail)
    jitter: float = 12.0                # max HSV value jitter (0-255) at detail=0
    sat_scale: tuple[float, ...] = (0.75, 0.85, 1.0, 1.0, 1.0)  # lean block-in, fat color

    # Wash (imprimatura) phase
    wash_radius: int = 128
    wash_alpha: float = 0.75
    wash_tone_pull: float = 0.35        # lerp toward the paper ground tone
    wash_desat: float = 0.3
    wash_max_points: int = 8

    # Sketch (underdrawing) phase
    sketch_radius: float = 1.0
    sketch_alpha: float = 0.55          # nothing in the sketch gets darker
    sketch_color: tuple[int, int, int] = (170, 166, 160)  # light gray — the
                                        # whole sketch is this color, per the
                                        # user's reference style
    sketch_length_budget: float = 5.0   # total line length, x image diagonal
    sketch_max_seg: float = 150.0       # split long chains so they draw over frames
    sketch_passes: tuple[int, int] = (1, 2)  # restated passes per contour (min, max)
    sketch_jitter: float = 2.0          # px of hand wobble on the first pass
    sketch_simplify: float = 10.0       # approxPolyDP epsilon (x scale): long
                                        # straight segments, not contour tracing
    sketch_erase_rate: float = 0.05     # chance a line is drawn wrong, undone, redrawn


    # Highlight phase
    highlight_count: int = 15
    highlight_radius: float = 2.0
    highlight_detail: float = 0.6
    highlight_min_accent: float = 60.0  # brightness-above-surroundings floor;
                                        # high on purpose — accents are rare,
                                        # and on texture-heavy subjects they
                                        # must stay inconspicuous

    # Timelapse
    fps: int = 30
    build_seconds: float = 20.0
    hold_seconds: float = 1.0
    ease_power: float = 2.2
    phase_fractions: tuple[float, ...] = (0.18, 0.08, 0.64, 0.10)  # sketch/wash/paint/highlight
    coherence_cells: int = 5            # spatial cells per axis for region-at-a-time order

    @property
    def debug_dir(self) -> Path:
        return self.out_dir / "debug"

    @property
    def cache_dir(self) -> Path:
        # Global, not per-output-dir: model outputs depend only on the input
        # image and working size, so a new -o dir shouldn't re-run models.
        return Path.home() / ".cache" / "painterly"
