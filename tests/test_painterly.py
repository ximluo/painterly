"""Fast invariant tests — no models, no rendering. Run:
.venv/bin/python -m unittest discover tests
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from painterly.config import Config
from painterly.maps import _cached
from painterly.order import painter_order
from painterly.render import _resample
from painterly.strokes import HIGHLIGHT, PAINT, SKETCH, WASH, Stroke, _bezier
from painterly.timelapse import stroke_schedule


def _stroke(seq, phase=PAINT, layer=1, bucket=0, face_id=0, in_face=False,
            in_eye=False, xy=(10.0, 10.0)):
    return Stroke(points=np.array([xy, (xy[0] + 5, xy[1] + 5)], np.float32),
                  radius=4.0, color=np.zeros(3, np.float32), alpha=1.0,
                  bucket=bucket, layer=layer, seq=seq, seed_near=0.5,
                  phase=phase, in_face=in_face, in_eye=in_eye, face_id=face_id)


class TestPainterOrder(unittest.TestCase):
    def _strokes(self):
        top = 2
        return [
            _stroke(0, phase=SKETCH, layer=0),
            _stroke(1, phase=WASH, layer=0),
            _stroke(2, layer=0, bucket=top, xy=(50, 50)),          # subject base
            _stroke(3, layer=0, bucket=0, xy=(5, 5)),              # bg base
            _stroke(4, layer=2, bucket=top, face_id=1, in_face=True,
                    xy=(50, 40)),                                  # face
            _stroke(5, layer=2, bucket=top, xy=(50, 70)),          # body
            _stroke(6, layer=2, bucket=0, xy=(5, 90)),             # background
            _stroke(7, phase=HIGHLIGHT, layer=5, bucket=top, xy=(52, 42)),
        ]

    def test_permutation_and_phase_order(self):
        for flow in ("one-go", "v11", "together"):
            with self.subTest(flow=flow):
                strokes = self._strokes()
                out = painter_order(strokes, 100, 100, cells=2,
                                    rng=np.random.default_rng(0),
                                    n_layers=5, face_flow=flow)
                self.assertEqual(sorted(s.seq for s in out),
                                 list(range(len(strokes))))
                phases = [s.phase for s in out]
                self.assertEqual(phases, sorted(phases))

    def test_subject_base_precedes_face(self):
        out = painter_order(self._strokes(), 100, 100, cells=2,
                            rng=np.random.default_rng(0))
        pos = {s.seq: i for i, s in enumerate(out)}
        self.assertLess(pos[2], pos[4])   # subject base before the face turn
        self.assertLess(pos[4], pos[5])   # face before its body
        self.assertLess(pos[5], pos[6])   # body before background

    def test_no_faces(self):
        strokes = [_stroke(0, layer=0, bucket=1, xy=(50, 50)),
                   _stroke(1, layer=2, bucket=1, xy=(50, 60)),
                   _stroke(2, layer=2, bucket=0, xy=(5, 5))]
        out = painter_order(strokes, 100, 100, cells=2,
                            rng=np.random.default_rng(0))
        pos = {s.seq: i for i, s in enumerate(out)}
        self.assertEqual(sorted(s.seq for s in out), [0, 1, 2])
        self.assertLess(pos[1], pos[2])   # subject detail before background


class TestStrokeSchedule(unittest.TestCase):
    CFG = Config(input_path=Path("x"), out_dir=Path("y"))

    def test_monotone_and_complete(self):
        strokes = ([_stroke(i, phase=SKETCH) for i in range(5)]
                   + [_stroke(i, phase=WASH) for i in range(5, 8)]
                   + [_stroke(i) for i in range(8, 58)]
                   + [_stroke(i, phase=HIGHLIGHT) for i in range(58, 60)])
        cum = stroke_schedule(strokes, 100, self.CFG)
        self.assertEqual(len(cum), 100)
        self.assertEqual(cum[-1], len(strokes))
        self.assertEqual(cum, sorted(cum))

    def test_empty_phases_inherit(self):
        strokes = [_stroke(i) for i in range(10)]  # paint only
        cum = stroke_schedule(strokes, 50, self.CFG)
        self.assertEqual(len(cum), 50)
        self.assertEqual(cum[-1], 10)

    def test_zero_frames(self):
        strokes = [_stroke(0)]
        self.assertEqual(stroke_schedule(strokes, 0, self.CFG)[-1], 1)


class TestGeometry(unittest.TestCase):
    def test_bezier(self):
        pts = np.array([[0, 0], [5, 5], [10, 0], [15, 5]], np.float32)
        out = _bezier(pts)
        self.assertEqual(out.shape, (10, 2))
        np.testing.assert_allclose(out[0], pts[0], atol=1e-5)
        np.testing.assert_allclose(out[-1], pts[-1], atol=1e-5)
        short = np.array([[0, 0], [1, 1]], np.float32)
        self.assertIs(_bezier(short), short)  # too short: passthrough

    def test_resample(self):
        pts = np.array([[0, 0], [10, 0], [20, 0]], np.float32)
        centers, tangents, ts = _resample(pts, spacing=2.0)
        self.assertEqual(len(centers), len(tangents))
        self.assertEqual(len(centers), len(ts))
        self.assertAlmostEqual(float(ts[0]), 0.0)
        self.assertAlmostEqual(float(ts[-1]), 1.0)
        # degenerate single point
        centers, tangents, ts = _resample(pts[:1], spacing=2.0)
        self.assertEqual(len(centers), 1)


class TestMapCache(unittest.TestCase):
    def test_content_key_invalidates(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            img = td / "input.jpg"
            img.write_bytes(b"first contents")
            cfg = Config(input_path=img, out_dir=td / "out")
            with patch.object(Config, "cache_dir",
                              property(lambda self: td / "cache")):
                calls = []
                first = _cached(cfg, "t", lambda: (calls.append(1),
                                                  np.ones(3))[1])
                _cached(cfg, "t", lambda: (calls.append(1), np.ones(3))[1])
                self.assertEqual(len(calls), 1)  # second call served from cache
                img.write_bytes(b"different contents")
                _cached(cfg, "t", lambda: (calls.append(1), np.zeros(3))[1])
                self.assertEqual(len(calls), 2)  # content change recomputes
                np.testing.assert_array_equal(first, np.ones(3))


if __name__ == "__main__":
    unittest.main()
