#!/bin/sh
set -eu

project=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build=$(mktemp -d)
trap 'rm -f "$build/uvc-exposure"; rmdir "$build"' EXIT

xcrun clang -std=c11 -Wall -Wextra -framework IOKit -framework CoreFoundation \
    "$project/webcams/uvc-exposure-macos.c" -o "$build/uvc-exposure"

# A4Tech cameras on ports 3 and 4 of this Mac's USB hub.
if [ "$#" -eq 0 ]; then set -- 00130000 00140000; fi
"$build/uvc-exposure" "$@"
