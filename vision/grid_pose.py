"""Estimate new camera poses from the blue 5 mm board and known lens parameters."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from stereo_scan import Camera, check_source_images, read_image


def grid_correspondences(image, camera, anchors, anchor_xy, bounds, spacing):
    size = np.array(image.shape[1::-1])
    anchors = np.asarray(anchors, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(anchors).all() or (anchors < 0).any() or (anchors >= size).any():
        raise ValueError("The four board anchors must be in the image.")
    anchors = cv2.undistortPoints(anchors, camera.matrix, camera.distortion, P=camera.matrix).reshape(-1, 2)
    xmin, xmax, ymin, ymax = bounds
    px_per_mm = 5
    origin = np.array([xmin, ymax])
    scale = np.array([px_per_mm, -px_per_mm])
    target = (anchor_xy - origin) * scale
    raster_size = (round((xmax - xmin) * px_per_mm) + 1, round((ymax - ymin) * px_per_mm) + 1)
    warp = cv2.getPerspectiveTransform(anchors.astype(np.float32), target.astype(np.float32))
    if abs(np.linalg.det(warp)) < 1e-12:
        raise ValueError("Board anchors must span a nondegenerate quadrilateral.")
    undistorted = cv2.undistort(image, camera.matrix, camera.distortion)
    board = cv2.warpPerspective(undistorted, warp, raster_size)
    blue = board[..., 0].astype(np.float32) - board[..., 2].astype(np.float32)

    xx, yy = np.meshgrid(
        np.arange(np.ceil(xmin / spacing), np.floor(xmax / spacing) + 1) * spacing,
        np.arange(np.ceil(ymin / spacing), np.floor(ymax / spacing) + 1) * spacing,
    )
    field_xy = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    guess = ((field_xy - origin) * scale).astype(np.float32)
    radius = round(.42 * spacing * px_per_mm)

    def ridge(profile):
        peak = int(profile.argmax())
        if peak in (0, len(profile) - 1) or profile[peak] - np.median(profile) < 2.3:
            return None
        a, b, c = profile[peak - 1:peak + 2]
        curve = a - 2 * b + c
        return peak + (.5 * (a - c) / curve if curve else 0)

    def intersection(pixel):
        x, y = np.rint(pixel).astype(int)
        if not (radius <= x < blue.shape[1] - radius and radius <= y < blue.shape[0] - radius):
            return (np.nan, np.nan)
        patch = blue[y-radius:y+radius+1, x-radius:x+radius+1]
        dx, dy = ridge(patch.mean(axis=0)), ridge(patch.mean(axis=1))
        if dx is None or dy is None:
            return (np.nan, np.nan)
        return (x - radius + dx, y - radius + dy)

    for _ in range(3):
        measured = np.array([intersection(pixel) for pixel in guess], dtype=np.float32)
        visible = np.isfinite(measured).all(axis=1)
        if visible.sum() < 40:
            raise ValueError("Too few blue grid intersections. Check the board anchor locations.")
        H, inliers = cv2.findHomography(field_xy[visible], measured[visible], cv2.RANSAC, 4.)
        if H is None or inliers.sum() < 40:
            raise ValueError("The grid intersections do not agree with the marked anchors.")
        refined = cv2.perspectiveTransform(field_xy[None], H)[0]
        if np.median(np.linalg.norm(refined - guess, axis=1)) > .5 * spacing * px_per_mm:
            raise ValueError("Grid numbering is ambiguous. Recheck the board anchor locations.")
        guess = refined

    keep = visible & (np.linalg.norm(measured - guess, axis=1) < 4.5)
    world = np.column_stack((field_xy[keep], np.zeros(keep.sum())))
    pixels = cv2.perspectiveTransform(measured[None, keep], np.linalg.inv(warp))[0]
    return world, pixels


def estimate_pose(image, lens, anchors, anchor_xy, bounds, spacing):
    world, pixels = grid_correspondences(image, lens, anchors, anchor_xy, bounds, spacing)
    ok, rotation, translation = cv2.solvePnP(world, pixels, lens.matrix, None)
    if not ok:
        raise ValueError("The visible board does not yield a consistent camera pose.")

    def residual(pose):
        projected, _ = cv2.projectPoints(world, pose[:3], pose[3:], lens.matrix, None)
        return (projected[:, 0] - pixels).ravel()

    # Fit points across the board rather than selecting a narrow strip through a warped sheet.
    fit = least_squares(residual, np.r_[rotation.ravel(), translation.ravel()],
                        loss="soft_l1", f_scale=3., x_scale="jac")
    error = np.linalg.norm(fit.fun.reshape(-1, 2), axis=1)
    ids = np.flatnonzero(error < 7.)
    if not fit.success or len(ids) < 40 or len(ids) < .6 * len(world):
        raise ValueError("The grid does not fit the lens calibration. Check board flatness, lens and anchors.")
    if np.any(np.ptp(world[ids, :2], axis=0) < .5 * np.ptp(anchor_xy, axis=0)):
        raise ValueError("The usable board corners cover too little of the field.")
    matrix = cv2.Rodrigues(fit.x[:3])[0]
    translation = fit.x[3:]
    if (-matrix.T @ translation)[2] <= 0:
        raise ValueError("The fitted camera is below the board. Check the anchor order.")
    return Camera(lens.matrix, lens.distortion, matrix, translation), {
        "grid_inliers": int(len(ids)), "median_error_px": float(np.median(error)),
        "p90_error_px": float(np.quantile(error, .9)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--lens", type=Path, required=True, help="existing calibration of the same lens and resolution")
    parser.add_argument("--annotations", type=Path, required=True, help="grid anchor positions for the new pair")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        paths = (args.left, args.right)
        lens_data = json.loads(args.lens.read_text())
        annotations = json.loads(args.annotations.read_text())
        names = [path.name for path in paths]
        check_source_images(annotations, paths, "Grid annotations")
        anchor_xy = np.asarray(annotations["grid_anchor_xy_mm"], dtype=float).reshape(4, 2)
        bounds = np.asarray(annotations["field_bounds_mm"], dtype=float).reshape(4)
        spacing = float(annotations["grid_spacing_mm"])
        if (
            not np.isfinite(np.r_[anchor_xy.ravel(), bounds, spacing]).all()
            or np.any(bounds[[1, 3]] <= bounds[[0, 2]])
            or spacing < 1 or np.any(bounds[[1, 3]] - bounds[[0, 2]] > 1000)
            or len(annotations["views"]) != 2 or len(lens_data["cameras"]) != 2
        ):
            raise ValueError("Check grid spacing, field bounds and the two views.")
        estimates = []
        for path, config, view in zip(paths, lens_data["cameras"], annotations["views"]):
            image = read_image(path)
            if list(image.shape[1::-1]) != lens_data["image_size"]:
                raise ValueError("Image resolution must match the lens calibration.")
            estimates.append(estimate_pose(image, Camera.from_dict(config), view["grid_anchors"],
                                           anchor_xy, bounds, spacing))
            print(f"{path.name}: {estimates[-1][1]}")
        data = {
            "description": "Approximate camera poses fitted from this pair's blue board grid, using existing lens parameters.",
            "source_images": names, "image_size": lens_data["image_size"],
            "cameras": [camera.as_dict() for camera, _ in estimates],
            "grid_diagnostics": [metrics for _, metrics in estimates],
        }
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Saved {args.output}")
    except (ValueError, KeyError, OSError, cv2.error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
