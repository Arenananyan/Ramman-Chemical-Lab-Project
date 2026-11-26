"""Build a metric surface mesh from a calibrated pair of still images."""

import argparse
from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

from plane_sweep import height_grid


@dataclass
class Camera:
    matrix: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray  # Field coordinates -> camera coordinates
    translation: np.ndarray  # Millimeters

    @classmethod
    def from_dict(cls, data):
        camera = cls(
            np.array(data["matrix"], dtype=float).reshape(3, 3),
            np.array(data["distortion"], dtype=float).ravel(),
            np.array(data["rotation"], dtype=float).reshape(3, 3),
            np.array(data["translation_mm"], dtype=float).reshape(3),
        )
        if not all(np.isfinite(x).all() for x in vars(camera).values()):
            raise ValueError("Calibration contains non-finite values.")
        return camera

    def as_dict(self):
        return {
            "matrix": self.matrix.tolist(),
            "distortion": self.distortion.ravel().tolist(),
            "rotation": self.rotation.tolist(),
            "translation_mm": self.translation.ravel().tolist(),
        }


@dataclass
class Mesh:
    vertices_mm: np.ndarray
    colors_rgb: np.ndarray
    triangles: np.ndarray


def check_source_images(config, paths, label):
    expected = config.get("source_images")
    actual = [path.name for path in paths]
    if expected is not None and expected != actual:
        raise ValueError(
            f"{label} was saved for {', '.join(expected)}, in that order. "
            f"Use data for {', '.join(actual)}."
        )


def read_image(path):
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        sips = shutil.which("sips")
        if not sips:
            raise ValueError("Convert HEIC to PNG or JPEG before scanning on this system.")
        with TemporaryDirectory() as directory:
            converted = Path(directory) / "frame.png"
            result = subprocess.run(
                [sips, "-s", "format", "png", str(path), "--out", str(converted)],
                capture_output=True, check=False,
            )
            if result.returncode or not converted.is_file():
                raise ValueError(f"Cannot convert {path}.")
            return read_image(converted)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}.")
    return image


def rectified_views(left, right, cameras, bounds, width):
    first, second = cameras
    size = left.shape[1], left.shape[0]
    relative_rotation = second.rotation @ first.rotation.T
    relative_translation = second.translation - relative_rotation @ first.translation
    if np.linalg.norm(relative_translation) < 1:
        raise ValueError("Camera centers are too close together for stereo.")
    r_left, r_right, _, projection, *_ = cv2.stereoRectify(
        first.matrix, first.distortion, second.matrix, second.distortion,
        size, relative_rotation, relative_translation[:, None],
    )
    if abs(projection[1, 3]) > abs(projection[0, 3]):
        # StereoSGBM searches horizontally. Rotate a vertical stereo pair.
        turn = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        r_left, r_right = turn @ r_left, turn @ r_right
    rectifications = r_left, r_right
    baseline = (r_right @ relative_translation)[0]
    corners = np.array(list(product(*bounds)))
    normalized = []
    for camera, rectification in zip(cameras, rectifications):
        xyz = (corners @ camera.rotation.T + camera.translation) @ rectification.T
        if np.any(xyz[:, 2] <= 0):
            raise ValueError("The scan volume crosses a rectified camera's horizon. Reduce the bounds.")
        normalized.append(xyz[:, :2] / xyz[:, 2:])

    # Frame the working volume instead of the two full images, which may overlap only near the board.
    margin = 24
    x_span = max(np.ptp(coords[:, 0]) for coords in normalized)
    rows = np.concatenate([coords[:, 1] for coords in normalized])
    focal = min((width - 2 * margin) / x_span, (width - 2 * margin) / np.ptp(rows))
    height = int(math.ceil(focal * np.ptp(rows))) + 2 * margin
    cy = (height - focal * (rows.min() + rows.max())) / 2
    centers = [(width - focal * (xy[:, 0].min() + xy[:, 0].max())) / 2 for xy in normalized]
    center_difference = centers[0] - centers[1]
    disparities = focal * (normalized[0][:, 0] - normalized[1][:, 0]) + center_difference
    minimum = math.floor(disparities.min()) - 4
    maximum = math.ceil(disparities.max()) + 4
    count = math.ceil((maximum - minimum + 1) / 16) * 16
    if count > 2 * width:
        raise ValueError("Disparity search is too wide. Reduce the scan volume or camera separation.")

    # OpenCV reserves columns at the sides for its full disparity search.
    pad_left = max(0, minimum + count)
    pad_right = max(0, -minimum)
    padded_size = (width + pad_left + pad_right, height)
    images, masks = [], []
    for image, camera, rectification, cx in zip((left, right), cameras, rectifications, centers):
        projection = np.array([[focal, 0., cx + pad_left], [0., focal, cy], [0., 0., 1.]])
        x_map, y_map = cv2.initUndistortRectifyMap(
            camera.matrix, camera.distortion, rectification, projection, padded_size, cv2.CV_32FC1,
        )
        images.append(cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR))
        masks.append(
            (x_map >= 2) & (x_map < size[0] - 3) & (y_map >= 2) & (y_map < size[1] - 3)
        )

    return images, masks, r_left, baseline, focal, centers[0], cy, center_difference, minimum, count, pad_left


def validate_scan(left, right, cameras, bounds_mm, max_edge_mm):
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise ValueError("Scan images must be same-size BGR images.")
    if len(cameras) != 2:
        raise ValueError("Exactly two calibrated cameras are required.")
    bounds = np.array(bounds_mm, dtype=float).reshape(3, 2)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Enter finite bounds in increasing X, Y and Z pairs.")
    if not math.isfinite(max_edge_mm) or max_edge_mm <= 0:
        raise ValueError("Max edge must be positive.")
    return bounds


def scan(left, right, cameras, bounds_mm, width=960, max_edge_mm=4.0):
    """Reconstruct matching surfaces in the field frame. Images are same-size BGR arrays."""
    bounds = validate_scan(left, right, cameras, bounds_mm, max_edge_mm)
    if not 320 <= width <= 1920:
        raise ValueError("Width must be 320–1920.")

    images, masks, rectification, baseline, f, cx, cy, offset, minimum, count, pad = (
        rectified_views(left, right, cameras, bounds, width)
    )
    block = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=minimum, numDisparities=count, blockSize=block,
        P1=8 * block**2, P2=32 * block**2, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=70, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    columns = slice(pad, pad + width)
    disparity = stereo.compute(*gray)[:, columns].astype(np.float32) / 16.0
    rows, cols = np.indices(disparity.shape)
    right_col = np.rint(cols + pad - disparity).astype(np.int32)
    right_valid = (right_col >= 0) & (right_col < masks[1].shape[1])
    right_valid &= masks[1][rows, right_col.clip(0, masks[1].shape[1] - 1)]
    denominator = disparity - offset
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -f * baseline / denominator
        rectified_xyz = np.stack(((cols - cx) * depth / f, (rows - cy) * depth / f, depth), axis=-1)
        first = cameras[0]
        xyz = (rectified_xyz @ rectification - first.translation) @ first.rotation

    valid = (disparity >= minimum) & (depth > 0) & masks[0][:, columns] & right_valid
    valid &= ((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1])).all(axis=-1)
    colors = images[0][:, columns, ::-1]
    return mesh_from_grid(xyz, colors, valid, max_edge_mm)


def scan_plane_sweep(left, right, cameras, bounds_mm, xy_step=.4, z_step=.25, patch_mm=2.4, max_edge_mm=4.0):
    """Return a measured surface mesh and the fraction of the XY grid in that mesh."""
    bounds = validate_scan(left, right, cameras, bounds_mm, max_edge_mm)
    centers = [-camera.rotation.T @ camera.translation for camera in cameras]
    if np.linalg.norm(centers[0] - centers[1]) < 1:
        raise ValueError("Camera centers are too close together for stereo.")
    corners = np.array(list(product(*bounds)))
    for camera in cameras:
        if np.any(corners @ camera.rotation[2] + camera.translation[2] <= 0):
            raise ValueError("The scan volume extends behind a camera.")
    xyz, colors, valid = height_grid(left, right, cameras, bounds, xy_step, z_step, patch_mm)
    if not valid.any():
        raise ValueError("No confident heights in this volume. Check calibration, texture and bounds.")
    mesh = mesh_from_grid(xyz, colors, valid, max_edge_mm)
    return mesh, len(mesh.vertices_mm) / valid.size


def mesh_from_grid(xyz, colors, valid, max_edge_mm):
    indices = np.full(valid.shape, -1, dtype=np.int32)
    indices[valid] = np.arange(valid.sum(), dtype=np.int32)
    vertices = xyz[valid]
    a, b = indices[:-1, :-1], indices[:-1, 1:]
    c, d = indices[1:, :-1], indices[1:, 1:]
    faces = np.concatenate(
        (np.stack((a, c, b), axis=-1).reshape(-1, 3),
         np.stack((b, c, d), axis=-1).reshape(-1, 3))
    )
    faces = faces[(faces >= 0).all(axis=1)]
    triangles = vertices[faces]
    edges = triangles - np.roll(triangles, 1, axis=1)
    faces = faces[(np.sum(edges**2, axis=2) <= max_edge_mm**2).all(axis=1)]
    if len(faces) == 0:
        raise ValueError("No connected stereo surface found in these bounds.")
    used, remapped = np.unique(faces, return_inverse=True)
    return Mesh(vertices[used], colors[valid][used], remapped.reshape(-1, 3))


def save_ply(mesh, output):
    """Write a binary PLY with vertex colors and metric field coordinates."""
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment field coordinates in millimeters, Z up\n"
        f"element vertex {len(mesh.vertices_mm)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(mesh.triangles)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    vertices = np.empty(len(mesh.vertices_mm), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    vertices["xyz"] = mesh.vertices_mm
    vertices["rgb"] = mesh.colors_rgb
    faces = np.empty(len(mesh.triangles), dtype=[("count", "u1"), ("indices", "<i4", 3)])
    faces["count"] = 3
    faces["indices"] = mesh.triangles
    with Path(output).open("wb") as file:
        file.write(header.encode("ascii"))
        vertices.tofile(file)
        faces.tofile(file)


def charuco(args):
    cols, rows = args.squares
    if not (4 <= cols <= 12 and 4 <= rows <= 12 and 0 < args.marker_mm < args.square_mm):
        raise ValueError("Use 4–12 squares per side and a positive marker size smaller than the square.")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard((cols, rows), args.square_mm, args.marker_mm, dictionary)


def make_target(args):
    board = charuco(args)
    cols, rows = args.squares
    image = board.generateImage((cols * 120 + 60, rows * 120 + 60), marginSize=30)
    args.output.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    print(f"Print the checker pattern at {cols * args.square_mm:g} x {rows * args.square_mm:g} mm (excluding border).")


def calibrate(args):
    if len(args.pair) < 8:
        raise ValueError("Provide at least eight calibration pairs with varied target positions and tilts.")
    field_origin = np.asarray(args.field_origin_mm, dtype=np.float64)
    if not np.isfinite(field_origin).all():
        raise ValueError("The target origin in bed coordinates must be finite.")
    board = charuco(args)
    detector = cv2.aruco.CharucoDetector(board)
    corners_world = board.getChessboardCorners().copy()
    corners_world[:, 1] *= -1  # Make +Z point away from the printed face.
    image_size = None

    def detect(path):
        nonlocal image_size
        image = read_image(path)
        size = image.shape[1], image.shape[0]
        if image_size is None:
            image_size = size
        if size != image_size:
            raise ValueError(f"{path} has resolution {size}, expected {image_size}.")
        corners, ids, *_ = detector.detectBoard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if ids is None or len(ids) < 10:
            raise ValueError(f"Fewer than ten ChArUco corners found in {path}.")
        return ids.ravel(), corners.reshape(-1, 2)

    world_points, left_points, right_points = [], [], []
    for left_path, right_path in args.pair:
        left_ids, left_xy = detect(left_path)
        right_ids, right_xy = detect(right_path)
        ids, left_index, right_index = np.intersect1d(
            left_ids, right_ids, assume_unique=True, return_indices=True,
        )
        if len(ids) < 10 or np.linalg.matrix_rank(np.cov(corners_world[ids, :2].T)) < 2:
            raise ValueError(f"Insufficient shared ChArUco corners in pair {left_path}, {right_path}.")
        world_points.append(corners_world[ids])
        left_points.append(left_xy[left_index])
        right_points.append(right_xy[right_index])

    left_error, k1, d1, *_ = cv2.calibrateCamera(world_points, left_points, image_size, None, None)
    right_error, k2, d2, *_ = cv2.calibrateCamera(world_points, right_points, image_size, None, None)
    stereo_error, _, _, _, _, relative_r, relative_t, *_ = cv2.stereoCalibrate(
        world_points, left_points, right_points, k1, d1, k2, d2, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    left_ids, left_xy = detect(args.field[0])
    right_ids, right_xy = detect(args.field[1])
    # For the flat field pair, the target's top-left pattern corner can be
    # offset from the bed origin. Target +X is right and +Y is up the pattern.
    field_corners = corners_world + field_origin
    ok, r1, t1 = cv2.solvePnP(field_corners[left_ids], left_xy, k1, d1)
    if not ok:
        raise ValueError("Cannot locate the flat field target.")
    rotation = cv2.Rodrigues(r1)[0]
    left = Camera(k1, d1, rotation, t1.ravel())
    right = Camera(k2, d2, relative_r @ rotation, (relative_r @ t1 + relative_t).ravel())
    checks = []
    for camera, ids, observed in ((left, left_ids, left_xy), (right, right_ids, right_xy)):
        center = -camera.rotation.T @ camera.translation
        if center[2] <= field_origin[2]:
            raise ValueError("The field target must face the cameras.")
        projected, _ = cv2.projectPoints(
            field_corners[ids], cv2.Rodrigues(camera.rotation)[0],
            camera.translation, camera.matrix, camera.distortion,
        )
        checks.append(float(np.sqrt(np.mean(np.sum((projected[:, 0] - observed)**2, axis=1)))))
    errors = [left_error, right_error, stereo_error, *checks]
    if not all(math.isfinite(x) and x <= 3 for x in errors):
        raise ValueError(f"Calibration reprojection RMS too high (pixels): {np.round(errors, 2)}. Retake the pairs.")
    result = {
        "image_size": image_size,
        "cameras": [left.as_dict(), right.as_dict()],
        "field_target_origin_mm": field_origin.tolist(),
        "reprojection_rms_px": dict(zip(("left", "right", "stereo", "field_left", "field_right"), errors)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}; reprojection RMS in pixels: {np.round(errors, 2)}")


def run_scan(args):
    start = perf_counter()
    if args.reference_height_mm is not None:
        if args.method != "plane-sweep":
            raise ValueError("Height verification requires --method plane-sweep.")
        if (
            not math.isfinite(args.reference_height_mm)
            or not args.bounds[4] < args.reference_height_mm < args.bounds[5]
            or not math.isfinite(args.max_error_mm)
            or args.max_error_mm <= 0
            or not 0 < args.min_coverage <= 1
        ):
            raise ValueError("Use a finite reference inside the Z bounds, positive error limit and coverage in (0, 1].")
    data = json.loads(args.calibration.read_text())
    check_source_images(data, (args.left, args.right), "Calibration")
    cameras = tuple(Camera.from_dict(camera) for camera in data["cameras"])
    if len(cameras) != 2:
        raise ValueError("Calibration must contain exactly two cameras.")
    left, right = read_image(args.left), read_image(args.right)
    size = tuple(data["image_size"])
    if any(image.shape[1::-1] != size for image in (left, right)):
        raise ValueError(f"Both input images must match the calibrated resolution {size}.")
    decoded = perf_counter()
    if args.method == "plane-sweep":
        mesh, coverage = scan_plane_sweep(
            left, right, cameras, args.bounds, xy_step=args.xy_step_mm, z_step=args.z_step_mm,
            patch_mm=args.patch_mm, max_edge_mm=args.max_edge_mm,
        )
        print(f"Measured XY coverage: {coverage:.1%} ({len(mesh.vertices_mm):,} grid samples).")
    else:
        mesh = scan(left, right, cameras, args.bounds, args.width, args.max_edge_mm)
    save_ply(mesh, args.output)
    end = perf_counter()
    print(
        f"Saved {args.output}: {len(mesh.vertices_mm):,} vertices, {len(mesh.triangles):,} triangles. "
        f"Decode {decoded - start:.2f} s, reconstruct and export {end - decoded:.2f} s."
    )
    if args.reference_height_mm is not None:
        error = np.abs(mesh.vertices_mm[:, 2] - args.reference_height_mm)
        median, p95, maximum = np.quantile(error, [.5, .95, 1])
        print(f"Absolute Z error: median {median:.3f}, P95 {p95:.3f}, max {maximum:.3f} mm.")
        if maximum > args.max_error_mm or coverage < args.min_coverage or len(error) < 100:
            raise ValueError(
                f"Height check failed. Required: max error ≤ {args.max_error_mm:g} mm, "
                f"coverage ≥ {args.min_coverage:.0%}, and at least 100 samples."
            )
        print("Height check passed for this reference region.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def target_arguments(command):
        command.add_argument("--squares", nargs=2, type=int, default=(7, 5), metavar=("COLS", "ROWS"))
        command.add_argument("--square-mm", type=float, default=25)
        command.add_argument("--marker-mm", type=float, default=18)
        command.add_argument("--output", type=Path, required=True)

    target = commands.add_parser("target", help="generate a printable ChArUco calibration target")
    target_arguments(target)
    target.set_defaults(run=make_target)

    setup = commands.add_parser("calibrate", help="calibrate the two fixed cameras once")
    target_arguments(setup)
    setup.add_argument("--pair", nargs=2, action="append", required=True, metavar=("LEFT", "RIGHT"))
    setup.add_argument("--field", nargs=2, required=True, metavar=("LEFT", "RIGHT"))
    setup.add_argument("--field-origin-mm", type=float, nargs=3, default=(0, 0, 0),
                       metavar=("X", "Y", "Z"), help="bed coordinates of the flat target's top-left pattern corner")
    setup.set_defaults(run=calibrate)

    reconstruction = commands.add_parser("scan", help="reconstruct a colored PLY mesh from one pair")
    reconstruction.add_argument("left", type=Path)
    reconstruction.add_argument("right", type=Path)
    reconstruction.add_argument("--calibration", type=Path, required=True)
    reconstruction.add_argument("--bounds", nargs=6, type=float, required=True,
                                metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"))
    reconstruction.add_argument("--output", type=Path, required=True)
    reconstruction.add_argument("--method", choices=("sgbm", "plane-sweep"), default="sgbm")
    reconstruction.add_argument("--width", type=int, default=960, help="SGBM rectified width in pixels")
    reconstruction.add_argument("--max-edge-mm", type=float, default=4.0)
    reconstruction.add_argument("--xy-step-mm", type=float, default=.4, help="plane-sweep XY grid spacing")
    reconstruction.add_argument("--z-step-mm", type=float, default=.25, help="plane-sweep height sampling interval")
    reconstruction.add_argument("--patch-mm", type=float, default=2.4, help="plane-sweep matching patch width")
    reconstruction.add_argument("--reference-height-mm", type=float, help="independently measured flat reference")
    reconstruction.add_argument("--max-error-mm", type=float, default=.1, help="maximum absolute error on the reference")
    reconstruction.add_argument("--min-coverage", type=float, default=.9, help="minimum measured fraction of reference XY")
    reconstruction.set_defaults(run=run_scan)

    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, KeyError, OSError, cv2.error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
