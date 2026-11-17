# Ramman-Chemical Lab Project

Estimate the dimensions and position of an upright cylinder from a calibrated pair of images, and operate a Klipper-based 3D printer from a Mac. The vision and printer tools currently run separately.

## Vision

Install the Python dependencies and run the synthetic tests.

```sh
cd vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_cylinder_fit
```

Run `python cylinder_fit.py --help` for the interface. Supply a pair of images in calibration order, the pair's calibration and the field bounds in millimeters.

```sh
python cylinder_fit.py ../inputs/left.jpg ../inputs/right.jpg \
  --calibration calibration.json --field-bounds 0 235 0 235 \
  --output-dir ../reconstruction/scan-01
```

Replace the example bounds with your measured field bounds. This produces a PLY with the cylinder and field, a JSON with the measured parameters, and an image showing the projected edges. Use `stereo_scan.py target --help` and `stereo_scan.py calibrate --help` to calibrate a fixed pair. Use `grid_pose.py --help` to estimate new poses with a known lens and marked grid anchors. The included JSON examples refer to earlier captures. Supply your own images and calibration. The project has yet to demonstrate 0.1 mm physical accuracy.

## Manual printer control

From the repository root, run `./printer demo 105 100 5 60` for a simulated horizontal move.

On a Mac on the same local network as the printer, run `./printer connect` followed by the actual IP displayed on the printer. Enter the SSH password in Terminal and leave that tab open. In a second tab at the repository root, set `SSH_HOST` to the same address and run `./printer status`.

Use `./printer preview X Y Z [SPEED]` to check a straight move. Review the physical path before using `./printer move X Y Z [SPEED]` and confirming with `MOVE`. Coordinates are in printer millimeters and speed is in millimeters per second. The controller checks homing, travel limits, bed mesh and coordinate offsets. Horizontal moves can request up to 60 mm/s within firmware limits. Moves that change Z are capped at 30 mm/s.

The next integration step is to register the field origin and axes to the printer coordinate system and validate a clearance-aware path before moving around the top rim.

The bundled printer controller retains its [MIT license](printer-control/LICENSE).
