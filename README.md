# Ramman-Chemical Lab Project

Estimate the dimensions and position of an upright cylinder from a calibrated pair of images, and operate a Klipper-based 3D printer from a Mac. The vision and printer tools currently run separately.

## Vision

### Two live webcams

From the repository root, start the local camera page with `python3 webcams/serve.py`,
then open [localhost:8768](http://localhost:8768) in Safari or Chrome. Click
**Connect cameras** and allow camera access. The page prefers the two A4tech
webcams and offers a separate selector for each view. Close other apps using
the cameras if a device cannot start.

**Capture pair** freezes both previews and provides downloads for the combined
PNG and the original-size left and right PNGs. Capture times are approximate,
without hardware synchronization. Frames stay in the browser. Stop the cameras
with **Stop cameras**, and stop the local server with Ctrl-C in Terminal.

### Lab setup and reference coordinates

On the Linux PC with both A4Tech webcams connected, run
`python3 webcams/serve.py` from the repository root. Open
[localhost:8768](http://localhost:8768), click **Connect cameras**, then run
`./camera-exposure.sh` in another terminal from the repository root. The
script uses USB ports 1 and 2 on the lab PC and sets both cameras to manual
exposure (80) and gain 0. If a camera resets, run the script again after
reconnecting. On Debian or Ubuntu, install `v4l-utils` if needed. Check
`ls -l /dev/v4l/by-path/` if the USB paths differ on another host.

The saved eight-pair calibration is
[`vision/raman-a4tech-20261124.json`](vision/raman-a4tech-20261124.json).
Use images at **1024 × 576 per camera** with the same camera mounting,
framing, and focus. The first view shows the printer head assembly to the
right of the bed. The second shows it to the left. Check the image content
when assigning the views, since the browser can swap camera order. Supply
the file as `--calibration raman-a4tech-20261124.json` when running the
vision commands from `vision/`.

| Reference | Coordinates or height (mm) |
| --- | --- |
| Recorded base-case toolhead position in printer coordinates | X = 120.0, Y = 222.94, Z = 200.0 |
| Graph-paper origin used by the stereo calibration | X = 0, Y = 0 at the marked black dot |
| Graph-paper surface in the stereo frame | Z = 0 |
| Tall validation target, measured above the paper | 83 |
| Short validation target, measured above the paper | 27 |

The base-case position records a past printer setup. The printer's actual
homing position is a separate reference. Stereo Z is height above the paper.
Before converting it to printer Z, verify the stock nozzle's paper-surface
reference after homing. Measure the Raman probe's working-distance offset
when it is installed. Verify the bed registration and repeat a known-height
check if the bed pose or the camera framing changes.

The eight-pair fit measured the 27 mm target within about 2 mm in the
tested placements. A separate cup-rim check after rehoming gave 27.1 mm
from stereo and 27 mm by caliper. These checks apply to the tested points.

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
