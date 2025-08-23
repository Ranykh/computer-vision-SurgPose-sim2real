#!/usr/bin/env bash
set -euo pipefail
PROJ_DATA="${PROJ_DATA:-/datashare/project}"
echo "[verify] Using PROJ_DATA=$PROJ_DATA"
if [ ! -d "$PROJ_DATA" ]; then
  echo "ERROR: $PROJ_DATA not found" >&2; exit 1
fi
echo "---- Expected items ----"
ls -lah "$PROJ_DATA" || true
echo "camera.json:"; ls -lah "$PROJ_DATA"/camera.json || echo "missing camera.json"
echo "*.obj models:"; find "$PROJ_DATA" -type f -name "*.obj" | wc -l
echo "COCO backgrounds (sample):"; find "$PROJ_DATA" -type f -name "*.jpg" | head -n 5 || true
echo "Polyhaven HDRIs (sample):"; find "$PROJ_DATA" -type f -name "*.hdr" -o -name "*.exr" | head -n 5 || true
echo "Unlabeled video 4_2_24_A_1.mp4:"; ls -lah "$PROJ_DATA"/videos/4_2_24_A_1.mp4 || echo "missing video"
