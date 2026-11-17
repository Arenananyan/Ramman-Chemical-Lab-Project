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

## Raman scan pipeline

The Python 3.10+ scanner plans grids, runs simulated acquisitions, saves spectra
and dark references in SQLite, resumes interrupted scans, and produces offline
reports and CSV exports. The scanner core uses the Python standard library.

From the repository root, run a 25-position simulated scan:

```sh
python3 -m raman_scanner plan --machine profiles/machine.simulation.json --scan profiles/scan.demo.json
python3 -m raman_scanner run --machine profiles/machine.simulation.json --scan profiles/scan.demo.json --output runs/demo
python3 -m unittest discover -s tests -v
```

Open `runs/demo/report.html` to view the synthetic map and spectra.
Use `profiles/scan.full-bed.json` for a 2,025-position simulation at 5 mm pitch.
The output directory is local to each machine.

### Progress on 17 November 2026

Imported the Raman scanner, example profiles, and tests from the original
project. The lab team reports that the printer reaches camera-selected points
accurately in their setup. This import adds the scanner alongside that workflow.
The scanner still needs a connection to the lab's vision and SSH motion
workflow, plus a real Avantes acquisition driver. The included hardware
template targets the original Moonraker adapter. See the
[instrument bridge protocol](docs/INSTRUMENT_BRIDGE.md) for the driver interface.

Validation for this import covered 31 scanner tests and a completed 25-position
simulation. A Node.js report check exercised map selection, raw and corrected
signal display, and CSV download using a mocked browser document and canvas.
