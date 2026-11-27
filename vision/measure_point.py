"""Convert a matched pair of image points into stock-nozzle printer XYZ."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from stereo_scan import Camera


DEFAULT_REGISTRATION = Path(__file__).with_name("raman-printer-registration-20261127.json")


def measure_point(left_px, right_px, image_size, capture_y_mm, registration=DEFAULT_REGISTRATION):
    """Measure one point, using pixels local to each image in calibration order."""
    registration = Path(registration)
    profile = json.loads(registration.read_text())
    calibration = json.loads((registration.parent / profile["stereo_calibration"]).read_text())
    if tuple(image_size) != tuple(calibration["image_size"]):
        width, height = calibration["image_size"]
        raise ValueError(f"Expected {width} × {height} pixels per camera.")
    if not math.isclose(capture_y_mm, profile["capture_printer_y_mm"], rel_tol=0, abs_tol=0.005):
        raise ValueError(f"This registration expects capture at printer Y = {profile['capture_printer_y_mm']} mm.")

    pixels = np.asarray([left_px, right_px], dtype=np.float64)
    if pixels.shape != (2, 2) or not np.isfinite(pixels).all():
        raise ValueError("Provide one finite (x, y) pixel coordinate per camera.")
    if (pixels < 0).any() or (pixels >= image_size).any():
        raise ValueError("Pixel coordinates must fall within their respective camera images.")

    cameras = [Camera.from_dict(entry) for entry in calibration["cameras"]]
    if len(cameras) != 2:
        raise ValueError("The stereo calibration must contain exactly two cameras.")
    projection = [np.column_stack((camera.rotation, camera.translation)) for camera in cameras]
    normalized = [
        cv2.undistortPoints(pixel.reshape(1, 1, 2), camera.matrix, camera.distortion).reshape(2, 1)
        for camera, pixel in zip(cameras, pixels)
    ]
    point = cv2.triangulatePoints(*projection, *normalized).ravel()
    if not np.isfinite(point).all() or abs(point[3]) < 1e-12:
        raise ValueError("These pixels do not triangulate to a finite point.")
    stereo_xyz = point[:3] / point[3]
    reprojection_errors = []
    for camera, pixel in zip(cameras, pixels):
        if (camera.rotation @ stereo_xyz + camera.translation)[2] <= 0:
            raise ValueError("The point is behind a camera. Check the view order and pixel match.")
        projected, _ = cv2.projectPoints(
            stereo_xyz.reshape(1, 3), cv2.Rodrigues(camera.rotation)[0],
            camera.translation, camera.matrix, camera.distortion,
        )
        reprojection_errors.append(float(np.linalg.norm(projected.ravel() - pixel)))

    transform = np.asarray(profile["printer_xyz_from_stereo_xyz1"], dtype=np.float64)
    if transform.shape != (3, 4) or not np.isfinite(transform).all():
        raise ValueError("The printer registration must contain a finite 3 × 4 matrix.")
    printer_xyz = transform @ np.r_[stereo_xyz, 1.0]
    return {
        "printer_xyz_mm": dict(zip(("x", "y", "z"), printer_xyz.tolist())),
        "reprojection_error_px": dict(zip(("left", "right"), reprojection_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--right", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--image-size", type=int, nargs=2, required=True, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--capture-y", type=float, required=True, help="Reported printer Y at image capture, in mm")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    args = parser.parse_args()
    try:
        result = measure_point(args.left, args.right, args.image_size, args.capture_y, args.registration)
        print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
