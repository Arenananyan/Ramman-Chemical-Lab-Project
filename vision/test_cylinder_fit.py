"""Calibrated synthetic geometry and rejection checks for cylinder detection."""

import unittest

import cv2
import numpy as np

from cylinder_fit import View, fit_cylinder
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
