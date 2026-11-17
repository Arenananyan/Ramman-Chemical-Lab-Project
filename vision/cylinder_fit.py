"""Detect one upright circular cylinder in a calibrated image pair."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from case_fit import closed_mesh, project, with_field
from stereo_scan import Camera, check_source_images, read_image, save_ply

ANGLES = np.linspace(0, 2 * np.pi, 96, endpoint=False)
UNIT_CIRCLE = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))


@dataclass
class View:
    image: np.ndarray
    projection: np.ndarray
    eye: np.ndarray
    distance: np.ndarray
    foreground: np.ndarray | None = None
    component_stats: np.ndarray | None = None

    @classmethod
    def create(cls, image, camera):
        # Keep enough detail for rim detection while limiting the edge search cost.
        scale = min(1., 1600 / image.shape[1])
        size = tuple(np.rint(np.array(image.shape[1::-1]) * scale).astype(int))
        matrix = camera.matrix.copy()
        matrix[:2] *= scale
        matrix[:2, 2] += (scale - 1) / 2
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        image = cv2.undistort(image, matrix, camera.distortion)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 16, 38, L2gradient=True)
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return cls(image, matrix @ np.column_stack((camera.rotation, camera.translation)),
                   -camera.rotation.T @ camera.translation, distance)

    def project(self, xyz):
        projected = xyz @ self.projection[:, :3].T + self.projection[:, 3]
        return projected[:, :2] / projected[:, 2:]

    def distances(self, points):
        pixels = self.project(points).astype(np.float32).reshape(1, -1, 2)
        return cv2.remap(self.distance, pixels, None, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=100).ravel()


def suppress_board_edges(views, bounds):
    """Reject texture that agrees between views when projected onto Z=0."""
    blurred = [cv2.GaussianBlur(view.image, (0, 0), 4).astype(np.float32) for view in views]
    planes = [view.projection[:, [0, 1, 3]] for view in views]
    for index, view in enumerate(views):
        other = 1 - index
        homography = planes[index] @ np.linalg.inv(planes[other])
        size = view.image.shape[1::-1]
        aligned = cv2.warpPerspective(blurred[other], homography, size)
        delta = np.max(np.abs(blurred[index] - aligned), axis=2)
        delta = cv2.boxFilter(delta, -1, (15, 15))
        field = np.zeros(view.image.shape[:2], dtype=np.uint8)
        board = np.rint(view.project(field_outline(bounds))).astype(np.int32)
        cv2.fillConvexPoly(field, board, 1)
        if not field.any():
            raise ValueError("The calibrated field is outside an image.")
        board_color = np.median(blurred[index][field != 0], axis=0)
        contrast = np.max(np.abs(blurred[index] - board_color), axis=2)
        foreground = ((delta > 12) & (contrast > 20) & (field != 0)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        foreground[:] = 0
        for contour in contours:
            if cv2.contourArea(contour) >= 120:
                cv2.drawContours(foreground, [contour], -1, 1, cv2.FILLED)
        near_object = cv2.dilate(foreground, kernel) != 0
        keep = ((view.distance < .01) & (near_object | (field == 0))).astype(np.uint8)
        view.distance = cv2.distanceTransform(1 - keep, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
        if count > 1:
            view.foreground = labels
            view.component_stats = stats


def features(params, view):
    x, y, radius, height = params
    center = np.array((x, y))
    rim = np.column_stack((center + radius * UNIT_CIRCLE, np.full(len(ANGLES), height)))
    bearing = view.eye[:2] - center
    tangent_angle = np.arccos(np.clip(radius / max(np.linalg.norm(bearing), radius), 0, 1))
    azimuth = np.arctan2(bearing[1], bearing[0])
    angles = azimuth + np.array((-tangent_angle, tangent_angle))
    tips = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    sides = [np.column_stack((np.tile(tip, (24, 1)), np.linspace(.08, .92, 24) * height)) for tip in tips]
    front = np.linspace(azimuth - tangent_angle, azimuth + tangent_angle, 32)
    base = np.column_stack((center + radius * np.column_stack((np.cos(front), np.sin(front))), np.zeros(32)))
    return rim, *sides, base


def field_outline(bounds):
    xmin, xmax, ymin, ymax = bounds
    return np.array(((xmin, ymin, 0), (xmax, ymin, 0), (xmax, ymax, 0), (xmin, ymax, 0)))


def candidates(views, bounds, radius_range, height_range):
    xmin, xmax, ymin, ymax = bounds
    # A horizontal circular rim projects as an ellipse. For each trial height,
    # undo that projection in both views and seek circles in their shared edges.
    pitch = max(.5, max(xmax - xmin, ymax - ymin) / 650)
    size = (int(np.ceil((xmax - xmin) / pitch)) + 1, int(np.ceil((ymax - ymin) / pitch)) + 1)
    step = 2.
    levels = np.linspace(*height_range, int(np.ceil(np.ptp(height_range) / step)) + 1)
    for height in levels:
        maps = []
        for view in views:
            px, py, pz, offset = view.projection.T
            transform = np.column_stack((pitch * px, pitch * py, xmin * px + ymin * py + height * pz + offset))
            maps.append(cv2.warpPerspective(view.distance, transform, size,
                                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=100))
        support = np.clip(5 - np.maximum(*maps), 0, 5)
        edges = cv2.GaussianBlur(np.uint8(support * 51), (5, 5), .8)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1., 9 / pitch,
                                   param1=65, param2=19, minRadius=max(3, int(radius_range[0] / pitch)),
                                   maxRadius=int(np.ceil(radius_range[1] / pitch)))
        if circles is not None:
            for x, y, radius in circles[0][:35]:
                yield np.array((xmin + x * pitch, ymin + y * pitch, radius * pitch, height))


def metrics(params, views):
    results = []
    for view in views:
        geometry = features(params, view)
        rim, left, right, base = [view.distances(points) for points in geometry]
        overlap = 0.
        top_gap = 100.
        if view.foreground is not None:
            projected = np.rint(view.project(np.vstack((geometry[0], geometry[-1])))).astype(np.int32)
            hull = cv2.convexHull(projected)
            x, y, width, height = cv2.boundingRect(hull)
            if x >= 0 and y >= 0 and x + width <= view.image.shape[1] and y + height <= view.image.shape[0]:
                footprint = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(footprint, hull - np.array((x, y), dtype=np.int32), 1)
                labels = view.foreground[y:y+height, x:x+width][footprint != 0]
                areas = view.component_stats[:, cv2.CC_STAT_AREA]
                intersections = np.bincount(labels, minlength=len(areas))
                unions = len(labels) + areas - intersections
                ratios = intersections[1:] / unions[1:]
                component = int(np.argmax(ratios)) + 1
                overlap = float(ratios[component - 1])
                if overlap >= .50:
                    center = view.project(np.array(((params[0], params[1], 0), (params[0], params[1], params[3]))))
                    direction = center[1] - center[0]
                    length = np.linalg.norm(direction)
                    if length > 5:
                        direction /= length
                        rect = view.component_stats[component, :4]
                        bx, by, bw, bh = rect
                        rows, cols = np.nonzero(view.foreground[by:by+bh, bx:bx+bw] == component)
                        observed = (cols + bx) * direction[0] + (rows + by) * direction[1]
                        top_gap = float(abs(np.quantile(observed, .995) - (projected @ direction).max()))
                    else:
                        top_gap = 0.
        results.append({
            "rim_median_px": float(np.median(rim)),
            "rim_coverage": float(np.mean(rim <= 3.5)),
            "rim_quarter_min": float(min(np.mean(quarter <= 3.5) for quarter in np.split(rim, 4))),
            "side_coverage": [float(np.mean(side <= 5.)) for side in (left, right)],
            "base_coverage": float(np.mean(base <= 5.)),
            "silhouette_iou": overlap,
            "top_gap_px": top_gap,
        })
    return results


def quality(results):
    return sum(m["rim_coverage"] + .35 * sum(m["side_coverage"]) + .4 * m["base_coverage"]
               - .045 * m["rim_median_px"] + 3 * m["silhouette_iou"]
               - .065 * min(m["top_gap_px"], 30) for m in results)


def exterior_side_points(params, view):
    """Trace the two outer body edges, starting from foreground cross sections."""
    if view.foreground is None:
        return None
    x, y, _, height = params
    base, top = view.project(np.array(((x, y, 0), (x, y, height))))
    axis = top - base
    length = np.linalg.norm(axis)
    if length < 20:
        return None
    direction = axis / length
    normal = np.array((-direction[1], direction[0]))
    cx, cy = np.rint((top + base) / 2).astype(int)
    if not (0 <= cx < view.image.shape[1] and 0 <= cy < view.image.shape[0]):
        return None
    label = view.foreground[cy, cx]
    if label == 0:
        return None
    rows, cols = np.nonzero(view.foreground == label)
    pixels = np.column_stack((cols, rows)) - base
    axial, lateral = pixels @ direction, pixels @ normal
    samples = [[], []]
    offsets = np.arange(-7., 8.)
    for fraction in np.linspace(.43, .70, 9):
        section = lateral[np.abs(axial - fraction * length) < 2]
        if len(section) < 20:
            continue
        for side, edge in enumerate(np.quantile(section, [.015, .985])):
            probes = base + fraction * axis + (edge + offsets[:, None]) * normal
            distances = cv2.remap(view.distance, probes.astype(np.float32)[None], None, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=100).ravel()
            best = int(np.argmin(distances + .14 * np.abs(offsets)))
            if distances[best] < 1.6:
                samples[side].append(probes[best])
    if min(map(len, samples)) < 5:
        return None
    return [np.array(side) for side in samples]


def refine_exterior(params, views, lower, upper):
    observations = [exterior_side_points(params, view) for view in views]
    if any(points is None for points in observations):
        return params

    def residual(body):
        errors = []
        for view, observed in zip(views, observations):
            _, first, second, _ = features(np.r_[body, params[3]], view)
            lines = [view.project(side[[0, -1]]) for side in (first, second)]
            axis = view.project(np.array(((body[0], body[1], 0), (body[0], body[1], params[3]))))
            direction = axis[1] - axis[0]
            normal = np.array((-direction[1], direction[0]))
            lines.sort(key=lambda line: line.mean(axis=0) @ normal)
            for line, points in zip(lines, observed):
                delta = line[1] - line[0]
                offset = points - line[0]
                errors.append((delta[0] * offset[:, 1] - delta[1] * offset[:, 0]) / np.linalg.norm(delta))
        return np.concatenate(errors)

    fit = least_squares(residual, params[:3], bounds=(lower[:3], upper[:3]), loss="soft_l1", f_scale=2)
    if fit.success and np.quantile(np.abs(fit.fun), .9) < 5:
        return np.r_[fit.x, params[3]]
    return params


def fit_cylinder(views, bounds, radius_range, height_range):
    xmin, xmax, ymin, ymax = bounds
    lower = np.array((xmin, ymin, radius_range[0], height_range[0]))
    upper = np.array((xmax, ymax, radius_range[1], height_range[1]))
    suppress_board_edges(views, bounds)

    def inside(params):
        x, y, r, _ = params
        return xmin <= x - r and x + r <= xmax and ymin <= y - r and y + r <= ymax

    def residual(params):
        weights = (1., .65, .65, .65)
        return np.concatenate([view.distances(points) * weight
                               for view in views for points, weight in zip(features(params, view), weights)])

    ranked = []
    for seed in candidates(views, bounds, radius_range, height_range):
        if inside(seed):
            evidence = metrics(seed, views)
            if all(m["rim_coverage"] >= .30 and m["silhouette_iou"] >= .50 for m in evidence):
                ranked.append((quality(evidence), seed))
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    starts = []
    for _, seed in ranked:
        if all(np.linalg.norm(seed - old) > 3 for old in starts):
            starts.append(seed)
        if len(starts) >= 30:
            break

    solutions = []
    for seed in starts:
        # The edge distance image uses fixed-point sampling. Use a finite
        # difference large enough to resolve a change in projected position.
        fit = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                            loss="soft_l1", f_scale=2., diff_step=.002, max_nfev=60)
        p = fit.x
        evidence = metrics(p, views)
        if inside(p) and all(
            m["rim_coverage"] >= .68 and m["rim_quarter_min"] >= .42
            and min(m["side_coverage"]) >= .35 and m["base_coverage"] >= .35 for m in evidence
        ) and all(
            m["silhouette_iou"] >= .65 for m in evidence
        ):
            solutions.append((quality(evidence), p, evidence))

    solutions.sort(key=lambda entry: entry[0], reverse=True)
    if not solutions:
        raise ValueError("No cylinder has a consistent rim, both sides and base in both images.")
    score, params, evidence = solutions[0]
    for other_score, other, _ in solutions[1:]:
        # Nearby fits usually follow neighboring edges of the same thick rim.
        if np.linalg.norm(params - other) > 5 and score - other_score < .22:
            raise ValueError("Several distinct cylinders have similar evidence in the two images.")
    refined = refine_exterior(params, views, lower, upper)
    if inside(refined):
        refined_evidence = metrics(refined, views)
        if all(m["silhouette_iou"] >= .65 and m["rim_coverage"] >= .55 for m in refined_evidence):
            return refined, refined_evidence
    return params, evidence


def save_overlay(path, views, bounds, params=None, reason=None, originals=None):
    frames = []
    for index, view in enumerate(views):
        frame = (view.image if originals is None else originals[index][0]).copy()
        if originals is None:
            project_points = view.project
        else:
            camera = originals[index][1]
            project_points = lambda points: project(points, camera)
        thickness = max(2, round(frame.shape[1] / 800))
        outline = project_points(field_outline(bounds))
        cv2.polylines(frame, [np.rint(outline).astype(np.int32)], True,
                      (255, 170, 25), thickness, cv2.LINE_AA)
        if params is not None:
            for feature_index, points in enumerate(features(params, view)):
                pixels = np.rint(project_points(points)).astype(np.int32)
                cv2.polylines(frame, [pixels], feature_index == 0,
                              (30, 225, 255), thickness, cv2.LINE_AA)
        else:
            frame[view.distance < .1] = (80, 190, 255)
        frames.append(frame)
    image = np.hstack(frames)
    if reason:
        cv2.putText(image, reason, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Could not encode the comparison image.")
    path.write_bytes(encoded.tobytes())


def run(args):
    paths = args.left, args.right
    config = json.loads(args.calibration.read_text())
    check_source_images(config, paths, "Calibration")
    cameras = [Camera.from_dict(data) for data in config["cameras"]]
    images = [read_image(path) for path in paths]
    if len(cameras) != 2 or any(list(image.shape[1::-1]) != config["image_size"] for image in images):
        raise ValueError("Both images must match the resolution of the two calibrated cameras.")

    bounds, radii, heights = (np.asarray(value, dtype=float)
                               for value in (args.field_bounds, args.radius_range, args.height_range))
    spans = bounds[[1, 3]] - bounds[[0, 2]]
    if (not np.isfinite(np.r_[bounds, radii, heights]).all() or np.any(spans <= 0)
            or not 1 <= radii[0] < radii[1] <= min(spans) / 2
            or not 1 <= heights[0] < heights[1] <= 300):
        raise ValueError("Enter increasing field, radius and height ranges in millimeters.")
    views = [View.create(image, camera) for image, camera in zip(images, cameras)]
    if np.linalg.norm(views[0].eye - views[1].eye) < 1:
        raise ValueError("Cylinder fitting requires two distinct calibrated positions.")
    for view in views:
        corners = field_outline(bounds)
        upper = corners + [0, 0, heights[1]]
        if np.any(np.vstack((corners, upper)) @ view.projection[2, :3] + view.projection[2, 3] <= 0):
            raise ValueError("The search volume extends behind a camera. Reduce the height range.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep each run together. A failed new scan must not look like an older successful scan.
    outputs = [args.output_dir / name for name in ("cylinder.ply", "measurements.json", "comparison.jpg")]
    if any(path.exists() for path in outputs):
        raise ValueError("Choose a fresh output folder for each scan.")
    mesh_path, measurements, overlay = outputs
    try:
        params, evidence = fit_cylinder(views, bounds, radii, heights)
    except ValueError as error:
        save_overlay(overlay, views, bounds, reason=str(error))
        raise
    x, y, radius, height = map(float, params)
    xy = UNIT_CIRCLE * radius + (x, y)
    rings = np.array([np.column_stack((xy, np.full(len(xy), z))) for z in (0., height)])
    save_overlay(overlay, views, bounds, params=params, originals=list(zip(images, cameras)))
    save_ply(with_field(closed_mesh(rings), bounds), mesh_path)
    data = {
        "source_images": [path.name for path in paths],
        "base_center_mm": [x, y, 0.], "top_center_mm": [x, y, height],
        "radius_mm": radius, "diameter_mm": 2 * radius, "height_mm": height,
        "field_bounds_mm": bounds.tolist(),
        "views": [dict(image=path.name, analysis_width_px=view.image.shape[1], **score)
                  for path, view, score in zip(paths, views, evidence)],
    }
    measurements.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Base center ({x:.2f}, {y:.2f}, 0) mm, diameter {2*radius:.2f} mm, height {height:.2f} mm.")
    print(f"Saved {mesh_path}, {measurements} and {overlay}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--field-bounds", type=float, nargs=4, required=True,
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--radius-range", type=float, nargs=2, default=(6, 40), metavar=("MIN", "MAX"))
    parser.add_argument("--height-range", type=float, nargs=2, default=(10, 120), metavar=("MIN", "MAX"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, OSError, cv2.error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
