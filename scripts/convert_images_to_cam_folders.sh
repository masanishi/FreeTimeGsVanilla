#!/usr/bin/env bash
set -euo pipefail

# Convert flat images like 0000_frame000000.png into folder layout:
# images/0000/000000.png
# images/0001/000000.png
#
# Usage:
#   ./scripts/convert_images_to_cam_folders.sh <images_dir>

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <images_dir>"
  exit 1
fi

IMAGES_DIR="$1"
if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "[ERROR] images_dir not found: $IMAGES_DIR"
  exit 1
fi

shopt -s nullglob
files=("$IMAGES_DIR"/*_frame*.png "$IMAGES_DIR"/*_frame*.jpg "$IMAGES_DIR"/*_frame*.jpeg)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "[INFO] No flat files found. If images are already in folders, nothing to do."
  exit 0
fi

moved=0
skipped=0

for f in "${files[@]}"; do
  base="$(basename "$f")"
  if [[ "$base" =~ ^([0-9]{4})_frame([0-9]{6})\.(png|jpg|jpeg)$ ]]; then
    cam="${BASH_REMATCH[1]}"
    frame="${BASH_REMATCH[2]}"
    ext="${BASH_REMATCH[3]}"

    mkdir -p "$IMAGES_DIR/$cam"
    dest="$IMAGES_DIR/$cam/$frame.$ext"

    mv "$f" "$dest"
    moved=$((moved + 1))
  else
    echo "[SKIP] Unrecognized name: $base"
    skipped=$((skipped + 1))
  fi

done

echo "[DONE] Moved: $moved files, Skipped: $skipped files"
