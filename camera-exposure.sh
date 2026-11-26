#!/usr/bin/env bash
set -e

for port in 1 2; do
  cam="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:${port}:1.0-video-index0"
  v4l2-ctl -d "$cam" --set-ctrl=auto_exposure=1
  v4l2-ctl -d "$cam" --set-ctrl=exposure_time_absolute=80,gain=0
  v4l2-ctl -d "$cam" --get-ctrl=auto_exposure,exposure_time_absolute,gain
done