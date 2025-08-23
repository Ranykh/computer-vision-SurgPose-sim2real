#!/usr/bin/env bash
set -euo pipefail
export PROJ_DATA="${PROJ_DATA:-/datashare/project}"
export PROJ_OUT="${PROJ_OUT:-$(pwd)/outputs}"
mkdir -p "$PROJ_OUT"

bash scripts/verify_resources.sh

python src/synthetic_data_generator.py --out "$PROJ_OUT/synth" --count 1200 --save-gt
python src/train.py --data "$PROJ_OUT/synth" --cfg configs/model_yolov8_pose.yaml --out "$PROJ_OUT/train_yolo"
python src/video.py --weights "$PROJ_OUT/train_yolo/best.pt" --video "$PROJ_DATA/videos/4_2_24_A_1.mp4" --out "$PROJ_OUT/results_synthetic_only.mp4"
python src/refine_unsupervised.py --weights "$PROJ_OUT/train_yolo/best.pt" --video "$PROJ_DATA/videos/4_2_24_A_1.mp4" --out "$PROJ_OUT/refine"
python src/video.py --weights "$PROJ_OUT/refine/best.pt" --video "$PROJ_DATA/videos/4_2_24_A_1.mp4" --out "$PROJ_OUT/results_refined.mp4"
