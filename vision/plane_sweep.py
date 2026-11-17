"""Measure visible heights on a metric XY grid using calibrated plane sweep."""

import math

import cv2
import numpy as np


def height_grid(left, right, cameras, bounds, xy_step, z_step, patch_mm):
    """Return field XYZ, RGB and a mask of locally distinctive height estimates."""
    if not all(math.isfinite(value) and value > 0 for value in (xy_step, z_step, patch_mm)):
        raise ValueError("Grid spacing, height spacing, and patch size must be positive.")

    nx, ny = np.floor(np.ptp(bounds[:2], axis=1) / xy_step + 1e-9).astype(int) + 1
    nz = math.ceil(np.ptp(bounds[2]) / z_step) + 1
    if min(nx, ny) < 3 or nz < 3:
        raise ValueError("Scan volume needs at least three grid samples in X, Y and Z.")
    if nx * ny > 2_000_000 or nx * ny * nz > 500_000_000:
        raise ValueError("Plane sweep grid is too large. Reduce the bounds or increase the spacing.")

    xmin, ymax = bounds[0, 0], bounds[1, 1]
    x, y = np.meshgrid(xmin + np.arange(nx) * xy_step, ymax - np.arange(ny) * xy_step)
    size = (int(nx), int(ny))
    projections, gray, visibility = [], [], []
    left_bgr = None
    for index, (image, camera) in enumerate(zip((left, right), cameras)):
        projection = camera.matrix @ np.column_stack((camera.rotation, camera.translation))
        projections.append(projection)
        maps = cv2.initUndistortRectifyMap(
            camera.matrix, camera.distortion, None, camera.matrix, image.shape[1::-1], cv2.CV_16SC2,
        )
        undistorted = cv2.remap(image, *maps, cv2.INTER_LINEAR)
        gray.append(cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY))
        full = np.full(image.shape[:2], 255, dtype=np.uint8)
        visibility.append(cv2.remap(full, *maps, cv2.INTER_LINEAR))
        if index == 0:
            left_bgr = undistorted

    patch = 2 * max(1, round(patch_mm / (2 * xy_step))) + 1
    if patch >= min(nx, ny):
        raise ValueError("Matching patch is larger than the scan area. Reduce --patch-mm.")

    def average(values):
        return cv2.boxFilter(values, -1, (patch, patch), borderType=cv2.BORDER_REFLECT101)

    def homography(projection, height):
        px, py, pz, offset = projection.T
        return np.column_stack((px * xy_step, -py * xy_step, px * xmin + py * ymax + pz * height + offset))

    def correlation(height):
        warps = [homography(projection, height) for projection in projections]
        a, b = [
            cv2.warpPerspective(image, warp, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP).astype(np.float32)
            for image, warp in zip(gray, warps)
        ]
        mean_a, mean_b = average(a), average(b)
        var_a = np.maximum(average(a * a) - mean_a**2, 0)
        var_b = np.maximum(average(b * b) - mean_b**2, 0)
        ncc = (average(a * b) - mean_a * mean_b) / np.sqrt(np.maximum(var_a * var_b, 1))
        seen = [
            cv2.warpPerspective(mask, warp, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
            for mask, warp in zip(visibility, warps)
        ]
        in_both = average(np.minimum(*seen).astype(np.float32)) > 254
        textured = (var_a >= 9) & (var_b >= 9)
        return np.where(in_both & textured, np.clip(ncc, -1, 1), -2)

    levels = np.linspace(*bounds[2], nz)
    dz = levels[1] - levels[0]
    best = np.full(x.shape, -2, dtype=np.float32)
    runner_up = best.copy()
    height = np.zeros_like(x)
    low, center = correlation(levels[0]), correlation(levels[1])
    for index, level in enumerate(levels[2:], start=2):
        high = correlation(level)
        peak = (center >= low) & (center > high) & (center > .1)
        better = peak & (center > best)
        runner_up = np.where(better, best, np.where(peak, np.maximum(runner_up, center), runner_up))
        best = np.where(better, center, best)

        # Fit a parabola through the sampled peak and its immediate neighbors.
        curvature = 2 * center - low - high
        adjustment = np.clip((high - low) / np.maximum(2 * curvature, 1e-6), -.5, .5)
        adjustment[(low < -1) | (high < -1)] = 0
        height[better] = (levels[index - 1] + dz * adjustment)[better]
        low, center = center, high

    valid = (best >= .82) & (best - runner_up >= .05)
    xyz = np.stack((x, y, height), axis=-1)
    # Sample color at the measured 3D coordinates rather than at a single reference plane.
    homogeneous = xyz @ projections[0][:, :3].T + projections[0][:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixel = (homogeneous[..., :2] / homogeneous[..., 2:]).astype(np.float32)
    colors = cv2.remap(left_bgr, pixel, None, cv2.INTER_LINEAR)[..., ::-1]
    return xyz, colors, valid
