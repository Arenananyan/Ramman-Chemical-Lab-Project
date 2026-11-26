"""Calibrated synthetic geometry and rejection checks for cylinder detection."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from case_fit import project
from cylinder_fit import View, field_outline, fit_cylinder
from stereo_scan import Camera


class CylinderFitTests(unittest.TestCase):
    bounds = (0., 200., 0., 200.)
    radii = (12., 30.)
    heights = (33., 70.)
    truth = np.array([76.3, 104.1, 20.2, 53.3])

    @classmethod
    def setUpClass(cls):
        intrinsics = np.array([[1150., 0, 400], [0, 1150., 300], [0, 0, 1]])
        cls.cameras = []
        for eye in (np.array([100., -245, 420]), np.array([160., 460, 390])):
            forward = np.array([95., 95, 20]) - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0, 0, 1])
            right /= np.linalg.norm(right)
            rotation = np.array((right, np.cross(forward, right), forward))
            cls.cameras.append(Camera(intrinsics, np.zeros(5), rotation, -rotation @ eye))

    def render(self, camera, params, cover_rim=False):
        image = np.full((600, 800, 3), 206, dtype=np.uint8)
        x, y, radius, height = params
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        xy = np.column_stack((x + radius * np.cos(angles), y + radius * np.sin(angles)))

        def ring(z):
            world = np.column_stack((xy, np.full(len(xy), z)))
            camera_xyz = world @ camera.rotation.T + camera.translation
            homogeneous = camera_xyz @ camera.matrix.T
            return np.rint(16 * homogeneous[:, :2] / homogeneous[:, 2:]).astype(np.int32)

        top, base = ring(height), ring(0)
        cv2.fillConvexPoly(image, cv2.convexHull(np.vstack((top, base))), (108, 116, 127),
                           cv2.LINE_AA, shift=4)
        cv2.fillConvexPoly(image, top, (51, 66, 85), cv2.LINE_AA, shift=4)
        if cover_rim:
            left, upper, width, length = cv2.boundingRect(top // 16)
            cv2.rectangle(image, (left - 10, upper - 10), (left + width + 10, upper + length + 10),
                          (206, 206, 206), cv2.FILLED)
        return image

    def fit(self, images):
        views = [View.create(image, camera) for image, camera in zip(images, self.cameras)]
        return fit_cylinder(views, self.bounds, self.radii, self.heights)[0]

    def test_known_geometry(self):
        found = self.fit([self.render(camera, self.truth) for camera in self.cameras])
        np.testing.assert_allclose(found, self.truth, rtol=0, atol=.6)

    def test_tall_off_center_cylinder_extends_above_projected_board(self):
        truth = np.array([160., 48., 26., 95.])
        self.heights = (10., 120.)
        images = []
        for camera in self.cameras:
            image = self.render(camera, truth)
            board = np.zeros(image.shape[:2], dtype=np.uint8)
            outline = np.rint(project(field_outline(self.bounds), camera)).astype(np.int32)
            cv2.fillConvexPoly(board, outline, 1)
            image[(board == 0) & (image == 206).all(axis=2)] = 153
            images.append(image)
        found = self.fit(images)
        np.testing.assert_allclose(found, truth, rtol=0, atol=.6)

    def test_short_cylinder_refinement_preserves_center(self):
        truth = np.array([105., 100., 20., 15.])
        self.heights = (10., 120.)
        found = self.fit([self.render(camera, truth) for camera in self.cameras])
        self.assertLess(np.linalg.norm(found[:2] - truth[:2]), .25)

    def test_cli_saves_errors_when_reference_exceeds_tolerance(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = [folder / f"{side}.png" for side in ("left", "right")]
            for path, camera in zip(paths, self.cameras):
                self.assertTrue(cv2.imwrite(str(path), self.render(camera, self.truth)))
            calibration = folder / "calibration.json"
            calibration.write_text(json.dumps({
                "image_size": [800, 600],
                "cameras": [camera.as_dict() for camera in self.cameras],
            }))
            output = folder / "output"
            python_flags = ["-s"] if sys.flags.no_user_site else []
            result = subprocess.run([
                sys.executable, *python_flags, str(Path(__file__).with_name("cylinder_fit.py")),
                *map(str, paths), "--calibration", str(calibration),
                "--field-bounds", "0", "200", "0", "200",
                "--radius-range", "12", "30", "--height-range", "33", "70",
                "--reference-dimensions-mm", "40.4", "53.3",
                "--reference-center-mm", "76.3", "104.1",
                "--max-error-mm", ".1", "--output-dir", str(output),
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Reference check failed", result.stderr)
            measurements = json.loads((output / "measurements.json").read_text())
            check = measurements["reference_check"]
            self.assertFalse(check["passed"])
            self.assertAlmostEqual(check["signed_errors_mm"]["diameter_mm"],
                                   measurements["diameter_mm"] - 40.4)
            self.assertLess(check["center_error_mm"], .25)
            self.assertTrue((output / "comparison.jpg").is_file())

    def test_rejects_empty_pair(self):
        with self.assertRaisesRegex(ValueError, "No cylinder"):
            self.fit([np.full((600, 800, 3), 206, dtype=np.uint8)] * 2)

    def test_rejects_covered_rim(self):
        with self.assertRaisesRegex(ValueError, "No cylinder"):
            self.fit([self.render(self.cameras[0], self.truth),
                      self.render(self.cameras[1], self.truth, cover_rim=True)])

    def test_rejects_moved_object(self):
        with self.assertRaisesRegex(ValueError, "No cylinder"):
            self.fit([self.render(self.cameras[0], self.truth),
                      self.render(self.cameras[1], self.truth + [38, -25, 0, 0])])


if __name__ == "__main__":
    unittest.main()
