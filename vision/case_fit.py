"""Fit a closed, rounded case from two calibrated views and marked contours."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from plane_sweep import height_grid
from stereo_scan import Camera, Mesh, check_source_images, read_image, save_ply, validate_scan


def project(points, camera):
    pixels, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        cv2.Rodrigues(camera.rotation)[0], camera.translation, camera.matrix, camera.distortion,
    )
    return pixels[:, 0]


def intersect_height(pixels, camera, height):
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera.matrix, camera.distortion)[:, 0]
    origin = -camera.rotation.T @ camera.translation
    rays = np.column_stack((normalized, np.ones(len(normalized)))) @ camera.rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (height - origin[2]) / rays[:, 2]
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("One lid outline does not intersect the top plane in front of its camera.")
    return (origin + rays * scale[:, None])[:, :2]


def contour_points(points, size):
    polygon = np.asarray(points, dtype=float)
    if (
        polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 8
        or not np.isfinite(polygon).all() or np.any(polygon < 0) or np.any(polygon >= size)
    ):
        raise ValueError("Each outline needs at least eight valid image points in order around the edge.")
    if cv2.contourArea(polygon.astype(np.float32)) < 25:
        raise ValueError("An outline encloses too little area.")
    return polygon


def mark_contour(image, title, previous=None):
    scale = min(1., 1200 / image.shape[1], 760 / image.shape[0])
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ratio = np.array(image.shape[1::-1]) / preview.shape[1::-1]
    clicks = []

    def mouse(event, x, y, flags, context):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setMouseCallback(title, mouse)
        while True:
            frame = preview.copy()
            if previous is not None:
                outline = np.rint(previous / ratio).astype(np.int32)
                cv2.polylines(frame, [outline], True, (190, 120, 20), 2, cv2.LINE_AA)
            if clicks:
                cv2.polylines(frame, [np.asarray(clicks, dtype=np.int32)], True, (30, 205, 255), 2, cv2.LINE_AA)
            cv2.imshow(title, frame)
            key = cv2.waitKey(35) & 0xFF
            if key in (10, 13) and len(clicks) >= 8:
                return contour_points(np.array(clicks) * ratio, np.asarray(image.shape[1::-1]))
            if key in (8, 127) and clicks:
                clicks.pop()
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) == 0:
                raise ValueError("Outline marking cancelled.")
    finally:
        cv2.destroyWindow(title)


def make_annotations(images):
    print("Click around each edge in order. At least eight points. Enter saves, right click undoes, Esc cancels.")
    result = []
    for side, image in zip(("Left", "Right"), images):
        outer = mark_contour(image, f"{side} - outside silhouette (exclude the shadow)")
        lid = mark_contour(image, f"{side} - flat lid edge where it starts to slope", outer)
        result.append({"silhouette": outer.tolist(), "lid": lid.tolist()})
    return {"views": result}


def top_height(images, cameras, lids, bounds):
    xyz, _, valid = height_grid(*images, cameras, bounds, xy_step=.35, z_step=.2, patch_mm=2.4)
    points = xyz[valid & (xyz[..., 2] > .7)]
    for image, camera, lid in zip(images, cameras, lids):
        if len(points) == 0:
            break
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(lid).astype(np.int32)], 1)
        mask = cv2.erode(mask, np.ones((11, 11), dtype=np.uint8))
        uv = np.rint(project(points, camera)).astype(int)
        w, h = image.shape[1::-1]
        visible = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        points, uv = points[visible], uv[visible]
        points = points[mask[uv[:, 1], uv[:, 0]] != 0]

    if len(points) < 25:
        raise ValueError("Too few matching lid points. Add visible texture or supply an independently measured height.")

    z = points[:, 2]
    ordered = np.sort(z)
    count = np.searchsorted(ordered, ordered + .6, side="right") - np.arange(len(ordered))
    first = int(np.argmax(count))
    cluster = ordered[first:first + count[first]]
    if len(cluster) < 25 or len(cluster) < .5 * len(z):
        raise ValueError("The lid does not have a dominant measured height. Check the outline and calibration.")
    height = float(np.median(cluster))
    print(f"Stereo lid Z: {height:.2f} mm, {len(cluster):,}/{len(z):,} clustered samples; "
          f"cluster P10–P90 {np.quantile(cluster, .1):.2f}–{np.quantile(cluster, .9):.2f} mm.")
    return height


def rotate(xy, angle):
    c, s = math.cos(angle), math.sin(angle)
    return xy @ np.array([[c, s], [-s, c]])


def fit_lid(xy):
    center, size, degrees = cv2.minAreaRect(xy.astype(np.float32))
    half_size = np.array(size) / 2
    if np.min(half_size) < .5:
        raise ValueError("Lid points are almost collinear or too small.")
    initial = np.r_[center, math.radians(degrees), np.log(half_size), .25]

    def residual(params):
        wh = np.exp(params[3:5])
        radius = min(wh) * params[5]
        local = np.abs(rotate(xy - params[:2], -params[2])) - wh + radius
        return np.linalg.norm(np.maximum(local, 0), axis=1) + np.minimum(local.max(axis=1), 0) - radius

    lower = [-np.inf, -np.inf, -np.inf, *np.log(half_size * .6), .015]
    upper = [np.inf, np.inf, np.inf, *np.log(half_size * 1.6), .85]
    result = least_squares(residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=.7)
    if not result.success:
        raise ValueError("Lid outline fit did not converge.")
    p = result.x
    wh = np.exp(p[3:5])
    print(f"Lid outline residual in its fitted plane: median {np.median(abs(result.fun)):.2f} mm.")
    return p[:2], p[2], wh, min(wh) * p[5]


def shell_rings(height, expansion, shoulder, center, angle, half_size, radius):
    # The unseen base uses a short bevel, with the bottom resting on the field.
    foot = min(.07 * height, .06 * min(half_size), radius)
    profiles = [(0, expansion - foot), (.04 * height, expansion - .3 * foot), (.10 * height, expansion)]
    for theta in np.linspace(0, np.pi / 2, 12):
        z = height * (shoulder + (1 - shoulder) * math.sin(theta))
        profiles.append((z, expansion * math.cos(theta)))

    directions = np.deg2rad(np.arange(4)[:, None] * 90 + np.linspace(0, 90, 24))
    normals = np.stack((np.cos(directions), np.sin(directions)), axis=-1)
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    rings = []
    for z, delta in profiles:
        r = radius + .65 * delta
        xy = ((half_size + delta - r) * signs[:, None] + r * normals).reshape(-1, 2)
        rings.append(np.column_stack((rotate(xy, angle) + center, np.full(len(xy), z))))
    return np.array(rings)


def fit_shell(height, lid_xy, silhouettes, cameras):
    center, angle, half_size, radius = fit_lid(lid_xy)
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    directions = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    observed = [(contour @ directions.T).max(axis=0) for contour in silhouettes]

    def residual(params):
        rings = shell_rings(height, *params, center, angle, half_size, radius)
        vertices = rings.reshape(-1, 3)
        return np.concatenate([
            (project(vertices, cam) @ directions.T).max(axis=0) - target
            for cam, target in zip(cameras, observed)
        ])

    result = least_squares(
        residual, [min(2., .1 * min(half_size)), .58],
        bounds=([0, .24], [.4 * min(half_size), .92]),
        loss="soft_l1", f_scale=4, diff_step=1e-4,
    )
    if not result.success:
        raise ValueError("Silhouette fit did not converge.")
    print(f"Silhouette residual in the input images: median {np.median(abs(result.fun)):.1f} px, "
          f"P95 {np.quantile(abs(result.fun), .95):.1f} px.")
    return shell_rings(height, *result.x, center, angle, half_size, radius), 2 * (half_size + result.x[0])


def closed_mesh(rings):
    layers, count = rings.shape[:2]
    vertices = np.vstack((rings.reshape(-1, 3), rings[0].mean(axis=0), rings[-1].mean(axis=0)))
    first_center, last_center = len(vertices) - 2, len(vertices) - 1
    a = np.arange(count)
    b = np.roll(a, -1)
    triangles = [np.column_stack((np.full(count, first_center), b, a))]
    for layer in range(layers - 1):
        lower = layer * count
        upper = lower + count
        triangles.extend((np.column_stack((lower + a, lower + b, upper + b)),
                          np.column_stack((lower + a, upper + b, upper + a))))
    triangles.append(np.column_stack((np.full(count, last_center), a + (layers - 1) * count,
                                      b + (layers - 1) * count)))
    color = np.tile([226, 226, 218], (len(vertices), 1)).astype(np.uint8)
    return Mesh(vertices, color, np.vstack(triangles))


def with_field(mesh, bounds):
    xmin, xmax, ymin, ymax = np.asarray(bounds, dtype=float).reshape(4)
    if not np.isfinite((xmin, xmax, ymin, ymax)).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError("Field bounds must be finite and increasing in X and Y.")
    field = np.array(((xmin, ymin, 0), (xmax, ymin, 0), (xmax, ymax, 0), (xmin, ymax, 0)))
    triangles = np.array(((0, 1, 2), (0, 2, 3))) + len(mesh.vertices_mm)
    return Mesh(
        np.vstack((mesh.vertices_mm, field)),
        np.vstack((mesh.colors_rgb, np.tile(np.uint8([204, 220, 231]), (4, 1)))),
        np.vstack((mesh.triangles, triangles)),
    )


def save_overlay(path, images, cameras, lids, silhouettes, rings):
    views = []
    for image, camera, lid, outline in zip(images, cameras, lids, silhouettes):
        model = project(rings.reshape(-1, 3), camera)
        outer = cv2.convexHull(model.astype(np.float32)).reshape(-1, 2)
        top = project(rings[-1], camera)
        uv = np.vstack((outline, outer, lid, top))
        low = np.maximum(uv.min(axis=0).astype(int) - 65, 0)
        high = np.minimum(uv.max(axis=0).astype(int) + 66, image.shape[1::-1])
        crop = image[low[1]:high[1], low[0]:high[0]]
        width = 700
        scale = width / crop.shape[1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for curve, color, thickness in (
            (outline, (210, 125, 35), 3), (lid, (210, 125, 35), 3),
            (outer, (30, 205, 245), 2), (top, (30, 205, 245), 2),
        ):
            points = np.rint((curve - low) * scale).astype(np.int32)
            cv2.polylines(crop, [points], True, color, thickness, cv2.LINE_AA)
        views.append(crop)
    height = max(len(view) for view in views)
    canvas = np.full((height + 62, 1420, 3), (39, 38, 36), dtype=np.uint8)
    for index, view in enumerate(views):
        x = index * 720
        canvas[42:42 + len(view), x:x + 700] = view
        cv2.putText(canvas, f"INPUT {index + 1}", (x + 14, 28), 0, .75, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Blue: traced edge      Yellow: 3D model projected into photo",
                (14, height + 56), 0, .53, (230, 230, 230), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(path.suffix, canvas)
    if not ok:
        raise ValueError("Could not encode the outline comparison.")
    path.write_bytes(encoded.tobytes())


def run(args):
    paths = (args.left, args.right)
    data = json.loads(args.calibration.read_text())
    check_source_images(data, paths, "Calibration")
    if not args.mark:
        annotations = json.loads(args.annotations.read_text())
        check_source_images(annotations, paths, "Case outlines")
    cameras = [Camera.from_dict(camera) for camera in data["cameras"]]
    images = [read_image(path) for path in paths]
    size = np.array(data["image_size"])
    if any(not np.array_equal(image.shape[1::-1], size) for image in images):
        raise ValueError("Images must match the calibration resolution.")
    bounds = validate_scan(*images, cameras, args.bounds, 4.)
    if args.mark:
        if args.annotations.exists():
            raise ValueError("Choose a new annotations file when using --mark.")
        annotations = make_annotations(images)
        annotations["source_images"] = [path.name for path in paths]
        if args.field_bounds is not None:
            annotations["field_bounds_mm"] = args.field_bounds
        args.annotations.write_text(json.dumps(annotations, indent=2) + "\n")
    views = annotations["views"]
    if len(views) != 2:
        raise ValueError("Annotations must contain exactly two views.")
    lids = [contour_points(view["lid"], size) for view in views]
    silhouettes = [contour_points(view["silhouette"], size) for view in views]
    height = args.height_mm
    if height is None:
        height = top_height(images, cameras, lids, bounds)
    elif not math.isfinite(height) or not max(0, bounds[2, 0]) < height < bounds[2, 1]:
        raise ValueError("The measured lid height must be positive and inside the Z search bounds.")
    else:
        print(f"Using supplied physical lid height: {height:.3f} mm.")
    lid_xy = np.vstack([intersect_height(lid, camera, height) for lid, camera in zip(lids, cameras)])
    rings, dimensions = fit_shell(height, lid_xy, silhouettes, cameras)
    mesh = closed_mesh(rings)
    field_bounds = args.field_bounds if args.field_bounds is not None else annotations.get("field_bounds_mm")
    if field_bounds is not None:
        mesh = with_field(mesh, field_bounds)
    save_ply(mesh, args.output)
    if args.overlay:
        save_overlay(args.overlay, images, cameras, lids, silhouettes, rings)
    length, width = sorted(dimensions, reverse=True)
    print(f"Saved {args.output}. Fitted length x width x height: {length:.2f} x {width:.2f} x {height:.2f} mm.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="outline JSON for the two images")
    parser.add_argument("--mark", action="store_true", help="mark outlines interactively and save the JSON")
    parser.add_argument("--bounds", type=float, nargs=6, required=True,
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"))
    parser.add_argument("--field-bounds", type=float, nargs=4,
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                        help="add the field plane at Z=0, with these XY limits in millimeters")
    parser.add_argument("--height-mm", type=float, help="independent physical measurement of the flat lid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, help="save a comparison of projected and traced outlines")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, OSError, cv2.error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
