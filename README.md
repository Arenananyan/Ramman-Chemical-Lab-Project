# Ramman-Chemical Lab Project

Estimate the dimensions and position of an upright cylinder from a calibrated pair of images, and operate a Klipper-based 3D printer from a Mac. The vision and printer tools currently run separately.

## Vision

### Two live webcams

From the repository root, start the local camera page with `python3 webcams/serve.py`,
then open [localhost:8768](http://localhost:8768) in Safari or Chrome. Click
**Connect cameras** and allow camera access. The page prefers the two A4tech
webcams and offers a separate selector for each view. Close other apps using
the cameras if a device cannot start.

Use **Digital exposure** to adjust both previews and their saved PNGs together.
The default of 100% leaves the image data unchanged.

The local server applies the hardware preset at startup and again after
**Connect cameras**. On this Mac, `bash camera-exposure-macos.sh` uses the two
A4Tech cameras at USB locations `00130000` and `00140000`. It uses Apple's
Command Line Tools and reads back manual mode, exposure 80, and gain 0.
If the cameras move to another hub or port, check `ioreg -p IOUSB -w0` and
update the location IDs in that script.

**Capture pair** freezes both previews and provides downloads for the combined
PNG and the original-size left and right PNGs. Capture times are approximate,
without hardware synchronization. Frames stay in the browser. Stop the cameras
with **Stop cameras**, and stop the local server with Ctrl-C in Terminal.

### Lab setup and reference coordinates

On the Linux PC with both A4Tech webcams connected, run
`python3 webcams/serve.py` from the repository root. Open
[localhost:8768](http://localhost:8768) and click **Connect cameras**. The
server runs `camera-exposure.sh` automatically. The
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

### Measure a marked point in printer coordinates

Use [the November 27 printer registration](vision/raman-printer-registration-20261127.json)
with the November 24 stereo calibration above. The registration expects the
same camera framing and the bed stopped at **printer Y = 222.94 mm during
capture**. The resulting Y is the target's printer coordinate.

Have Codex identify the **same physical mark** in both images and return its
pixel coordinates. Use the calibration camera order described above. For a
combined capture, split the two views first so each coordinate is measured
from its own image's top-left corner. Use **1024 × 576** per view. If you
resize the full views from the same framing, locate the pixels after resizing.

With the vision dependencies installed, call this from the repository root.
The example replays the 67 mm reference at the corner.

```sh
python3 vision/measure_point.py \
  --left 445.5 103 --right 408 109.5 \
  --image-size 1024 576 --capture-y 222.94
```

The command returns JSON with `printer_xyz_mm` and the per-view
`reprojection_error_px`. For this example, XYZ is approximately
`(164.55, 35.83, 66.75)` mm. Pass the numeric arguments as separate
subprocess arguments on the website backend. Read stdout only on success
and surface the error message on failure. If Codex cannot identify the
same mark in both views, have the operator select it.

The matrix includes the **1.04046 height scale** and **−0.04 mm paper
offset** for the stock nozzle. Two checks with the 67 mm target agreed
with nozzle contact within 0.25 mm in Z. Reprojection error describes the
pixel match and cannot establish physical accuracy by itself. The command
returns a surface estimate. Apply travel clearance and any Raman probe
offset separately in the motion workflow.

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

### Calibrate the webcams to the bed

The cameras stay fixed while you capture the ChArUco target. Capture at
least eight pairs with the target held still at different positions and tilts,
including across the image. Use **Capture pair** and save the individual left
and right PNGs from the webcam page. Keep the same camera order, resolution,
and focus for calibration and subsequent scans.

Capture one more pair with the target flat on the bed. Align its bottom edge
with bed +X and its left edge with bed +Y. The calibration command accepts the
bed coordinates of the **top-left pattern corner**, excluding the white margin.
With the bottom-left pattern corner at bed (0, 0), the default 7 × 5 target
puts the top-left at (0, 125) mm. Use the height of the printed face above the
bed as the third coordinate. If you place the target elsewhere, measure the
top-left corner's coordinates in the same bed frame and use them instead.

From the repository root, this example uses eight pairs saved as
`inputs/calibration/01-left.png` through `08-right.png`. Set the Z value to your
measured target thickness before running the command.

```bash
pairs=()
for i in {01..08}; do
  pairs+=(--pair "inputs/calibration/${i}-left.png" "inputs/calibration/${i}-right.png")
done
python3 vision/stereo_scan.py calibrate "${pairs[@]}" \
  --field inputs/calibration/bed-left.png inputs/calibration/bed-right.png \
  --field-origin-mm 0 125 0 --output inputs/calibration/bed-calibration.json
```

Print the target at actual size and verify the square spacing with a ruler or
caliper. The calibration estimates the camera positions and orientations from
the images. Its reprojection errors are in pixels. Use an independently
measured object to check how those errors translate into millimeters.

### Check accuracy on the lab setup

Capture the same stationary reference cylinder with both calibrated cameras.
Keep the calibration resolution, camera order, focus, and mounts fixed. Measure
the cylinder's diameter and height independently and supply the values when you
run the fit from `vision/`. For example, for a 40.4 mm diameter and 53.3 mm height:

```sh
python cylinder_fit.py ../inputs/left.png ../inputs/right.png \
  --calibration calibration.json --field-bounds 0 235 0 235 \
  --reference-dimensions-mm 40.4 53.3 --output-dir ../reconstruction/reference-01
```

Use the measured bounds of your own field. If you have an independently
measured center in the calibration frame, add `--reference-center-mm X Y`.
The CLI reports signed errors and writes them to `measurements.json`. Add
`--max-error-mm` with your acceptance tolerance for a pass/fail check of each
dimension and the XY center distance, when supplied. A failed check exits with
status 2 and keeps the measurements and overlay for inspection.

Capture several pairs at each tested position to check repeatability. Use
the center and different parts of the bed to check how error varies across the
field. Keep the reference measurements independent of the camera fit.

On 23 November 2026, all seven vision tests passed with the system Python
packages. The cylinder fitter accepted 13 of 13 rendered cases after the
foreground and refinement fixes, compared with 5 of 13 before. Largest errors
in the revised fits were 0.238 mm for XY center distance, 0.754 mm for diameter,
and 0.107 mm for height. These tests used known synthetic camera geometry.
Physical accuracy still needs measured reference captures from the lab.

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

## Arduino pump control

Upload [timed_pump.ino](pump-control/timed_pump/timed_pump.ino) to the Arduino
Nano (ATmega328P). The sketch drives the active-low pump output on D9 and
communicates over USB serial at 9600 baud. Close Arduino Serial Monitor
before connecting with Python. From the repository root:

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install -r pump-control/requirements.txt
./.venv/bin/python pump-control/pump.py --port /dev/cu.usbserial-110 off
```

Use the port shown in the Arduino IDE if the device name changes. `off` checks
communication by asking the Nano to switch the output off. With the tubing
aimed into a cup for a supervised water test, try a half-second pulse.

```sh
./.venv/bin/python pump-control/pump.py --port /dev/cu.usbserial-110 pulse 0.5
```

The Nano accepts pulses from 0.1 to 2 seconds, switches the output off when
its timer expires, and then reports completion. The timer runs even if the
USB serial link drops while the Nano stays powered. The reported completion
describes the output state. Measure the dispensed liquid separately.

## Experiment desk

Start the local server and open [localhost:8768](http://localhost:8768).
Select **Simulation**, click **Start demo**, and answer **yes** or **no** when
the coordinator asks about the head position. The workflow has Vision,
Motion, and Pump handlers, and uses the existing mocked printer in simulation.

```sh
./.venv/bin/python -m pip install -r vision/requirements.txt -r pump-control/requirements.txt
./.venv/bin/python webcams/serve.py
```

Set `GROQ_API_KEY` in the server's environment to enable natural-language
chat and the camera-to-motion handoff. The default model is
`openai/gpt-oss-20b`. Use `GROQ_MODEL` to select another Groq model with
strict JSON Schema support. Groq receives operator messages, run state, and
the numeric fit and target for the handoff. The webcam frames stay local.
For a fitted cylinder, Groq can propose positioning at the fixed target or
hold the run. The existing controller validates motion and requires the
operator's position approval before pumping. Without a key, the desk
accepts its original local commands.

For live use, verify the stereo calibration at 1024 × 576 with a measured
reference in the current camera mounting. The left selector must show the
head assembly to the right of the bed in the reference pose, matching the
first calibration view. The right selector must show the other view.

Create `profiles/experiment.local.json` using a measured camera-field to
printer registration. This template identifies the measurements to supply.

```json
{
  "verified_for_live": false,
  "field_bounds_mm": [null, null, null, null],
  "printer_xy_from_field": [[null, null, null], [null, null, null]],
  "printer_z_at_paper_mm": null,
  "clearance_z_mm": null
}
```

The bounds are field X min/max followed by Y min/max. Each transform row
multiplies field X, field Y, and 1 to give printer X or Y. Measure the XY
transform for the dispensing tip. The paper reference is the commanded
printer Z when that tip touches the paper. Choose a clearance Z whose
full XY travel path is clear. Verify the measured mapping and path before
setting `verified_for_live` to true.

Open the existing printer SSH connection and launch the webpage server
with `SSH_HOST` set to the printer address and `PRINTER_ALLOW_MOTION=1`.
The pump port defaults to `/dev/cu.usbserial-110` and can be set with
`PUMP_PORT` when launching the server.

Live mode fits one upright cylinder from raw stereo frames. The head moves
at 5 mm/s through the verified clearance and stops 10 mm above the fitted
top. The controller rechecks the position after approval, then requests a
single 0.5-second water pulse. The position approval expires after two minutes.
The input frames, fit, and overlay are saved under `runs/experiment/`.
