# painterly

Turns a photo into a painting, stroke by stroke, in the order an artist would paint it.

It plans thousands of brush strokes with classical graphics techniques and uses depth and saliency models to decide what to paint when: sketch, wash, subject before background, faces one at a time, detail patch by patch. You get the final painting plus a timelapse of every stroke, and the run is deterministic: same photo and seed gives the same painting, byte for byte.

<p align="center">
  <img src="docs/timelapse.gif" width="45%" alt="build-up timelapse" />
  &nbsp;
  <img src="docs/input_cat.jpg" width="34%" alt="source photo" />
</p>

## How it works

**1. Stroke engine: Hertzmann '98, modernized.** The core loop is Aaron Hertzmann's [*Painterly Rendering with Curved Brush Strokes of Multiple Sizes*](https://www.dgp.toronto.edu/papers/aherzmann_SIGGRAPH1998.pdf) (SIGGRAPH 1998). Five brush sizes paint coarse to fine, and each layer only places strokes where the canvas still differs from a blurred version of the photo.

**2. Stroke shape, direction, and texture.** Raw Hertzmann strokes wander around like worms. Three fixes:
- Strokes follow an **Edge Tangent Flow** field ([Kang, Lee & Chui, *Coherent Line Drawing*, NPAR 2007](https://dl.acm.org/doi/10.1145/1274871.1274878)), with the smoothing kernel scaled to the brush size, so nearby strokes stay parallel like real brushwork.
- Each stroke is a **single quadratic Bézier** with a cap on total turn (about 60°). This is the same stroke shape modern neural painters use ([Paint Transformer, ICCV 2021](https://arxiv.org/abs/2108.03798); [Stylized Neural Painting, CVPR 2021](https://arxiv.org/abs/2011.08114); [Learning to Paint, ICCV 2019](https://arxiv.org/abs/1903.04411)).
- Strokes render as **tapered, bristle-textured ribbons**: a procedural bristle texture warped along the stroke (the rendering trick from Stylized Neural Painting), with wet blending against the paint underneath.

**3. Pen pressure from handwriting research.** Width and opacity taper along each stroke with a lognormal velocity profile (Plamondon's kinematic theory of human movement), and stroke width responds to curvature via the [two-thirds power law](https://pubmed.ncbi.nlm.nih.gov/6666128/) (Lacquaniti, Terzuolo & Viviani, 1983). Fast straight strokes go thin and dry, slow curves press wider.

**4. Depth and saliency decide where the detail goes.** [Depth Anything V2](https://arxiv.org/abs/2406.09414) gives relative depth, and a [BiRefNet](https://arxiv.org/abs/2401.03407) matte gives subject saliency. Depth is split into paint groups (the subject always goes in the top group), so the background is painted far to near with big loose strokes while the subject gets the fine brushes. YuNet face detection with eye landmarks gives faces an extra fine pass, and the eyes get the finest brush.

**5. The painting order.** Based on [Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO), the [Time-Map digital painting dataset](https://cragl.cs.gmu.edu/timemap/), and portrait painting tutorials:
- a light gray **construction sketch**: long rough lines, drawn over a couple times with hand wobble, plus the occasional wrong line that gets erased. The sketch lives on an overlay and disappears only where paint covers it (a per-pixel cover map)
- a thin **wash** under the sketch
- the **subject's base coat before the background** gets anything past the wash
- **faces one at a time**, largest first, coarse to fine with the eyes leading, while rough block-in strokes land elsewhere so the canvas keeps filling in around them. The background's base only starts once the faces are mostly done
- bodies refined in **anatomical zones** (hair and head first, then down)
- small brushes finish **one patch before moving to the next**
- a few bright **highlights** at the very end

Finished faces are protected by soft masks, so later strokes can't smear them.

## From v1 to final

It took 23 tagged versions to get the process right. Same engine skeleton, same photo:

| v1 had the phases, but | final |
| --- | --- |
| strokes wandered like worms | flow-aligned Bézier ribbons with pressure taper |
| the sketch was a perfect contour trace | sparse rough lines, drawn over, with erases |
| detail showed up randomly everywhere | faces, then zones, then patches |
| hard seams between depth planes | feathered soft masks between paint groups |
| every region treated the same | detail gated by depth, saliency, and faces |

<p align="center">
  <img src="docs/timelapse_v1.gif" width="45%" alt="first version" />
  &nbsp;
  <img src="docs/timelapse.gif" width="45%" alt="final version" />
</p>

It works past portraits too, since the subject-first order and depth grouping come from the models:

<p align="center">
  <img src="docs/timelapse_swan.gif" width="45%" alt="swan timelapse" />
  &nbsp;
  <img src="docs/input_swan.jpg" width="33.6%" alt="swan source photo" />
</p>

## Usage

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run painterly examples/golden.jpg            # painting.png + timelapse.mp4 in out/golden/
uv run painterly photo.jpg -o out/mypainting --size 1080 --seed 3
```

Useful flags:

| flag | what it does |
| --- | --- |
| `--size N` | long-edge working resolution (default 1080) |
| `--seed N` | reroll the stroke plan |
| `--device auto\|mps\|cuda\|cpu` | depth model device (default `auto`: MPS > CUDA > CPU) |
| `--face-flow one-go\|v11\|together` | how faces are scheduled (default `one-go`) |
| `--no-video` | skip the timelapse, just render the painting |
| `--classic` | plain Hertzmann strokes, no flow field or ribbons |
| `--flat` | untextured flat strokes |

The first run downloads Depth Anything V2 (small) from Hugging Face. BiRefNet and YuNet weights are vendored or fetched once, and per-image model outputs are cached in `~/.cache/painterly`, so re-rendering the same photo skips straight to stroke planning. A 1080p painting with its 20-second timelapse takes about a minute on an M-series Mac. Everything besides the depth model runs on OpenCV + ONNX Runtime, so macOS, Linux, and Windows all work.

## Layout

```
src/painterly/
  maps.py       depth / saliency / face maps, caching
  strokes.py    stroke generation: layers, ETF, Bézier, face & eye passes
  sketch.py     construction-sketch extraction and human-style sketch lines
  order.py      the painting order
  render.py     ribbon rasterizer, pressure, masks, wet blending
  timelapse.py  stroke scheduling and video writer
  cli.py        entry point
```
