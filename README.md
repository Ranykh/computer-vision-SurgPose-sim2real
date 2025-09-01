# Synthetic-to-Real Pose Estimation for Surgical Tools

A reproducible pipeline for **2D pose estimation of surgical instruments** trained on **synthetic data** and refined on **unlabeled surgical video** (unsupervised domain adaptation).

> This repository implements the project requirements end-to-end:
> - **Phase 1** — Synthetic data generation (COCO + 5 keypoints per tool)  
> - **Phase 2** — Training on synthetic data only + video inference  
> - **Phase 3** — Unsupervised refinement on real video via pseudo-labels (self-training)

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data & Paths](#data--paths)
- [Repository Layout](#repository-layout)
- [Reproduce the Results (Step-by-Step)](#reproduce-the-results-step-by-step)
  - [Phase 1 — Synthetic Data Generation](#phase-1--synthetic-data-generation)
  - [Phase 2 — Train on Synthetic Only](#phase-2--train-on-synthetic-only)
  - [Phase 3 — Unsupervised Refinement (UDA)](#phase-3--unsupervised-refinement-uda)
  - [Inference (Images & Video)](#inference-images--video)
- [Pretrained Weights (Downloads)](#pretrained-weights-downloads)
- [Troubleshooting & Tips](#troubleshooting--tips)

---

## Environment Setup

**Tested:** Ubuntu 20.04/22.04 · Python 3.10+ · CUDA (any version compatible with your PyTorch build)  
**Rendering:** [BlenderProc](https://github.com/DLR-RM/BlenderProc) (invoked with `python -m blenderproc run ...`)

```bash
# 1) Clone and enter
git clone [https://github.com/Ranykh.git]
cd surg-pose-sim2real

# 2) Virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3) Install dependencies
pip install -r requirements.txt
```
## Data & Paths

The repo expects the course VM assets under `/datashare/project` (you can override via env vars below):

- `camera.json` — camera intrinsics  
- `surgical_tools_models/needle_holder/*.obj`, `surgical_tools_models/tweezers/*.obj`  
- `haven/hdris/` — HDR environments  
- `train2017/` — 2D background images from COCO 2017  
- `vids_test/4_2_24_A_1.mp4` — unlabeled surgical video

**Set  environment variables:**
```bash
export PROJ_DATA=/datashare/project         # where provided assets live
export PROJ_OUT=$(pwd)/outputs              # where this repo will write results
mkdir -p "$PROJ_OUT"
```
## Repository Layout

```text
computer-vision-SurgPose-sim2real/
├─ configs/                         # YAMLs for data/model/training (Ultralytics/YOLO)
├─ refine_v1_yolo8m_960/           # Phase 3 refined run artifacts (weights, logs, results)
├─ scripts/                         # Helper shell scripts (e.g., verify_resources.sh)
├─ src/
│  ├─ synthetic_data_generator.py   # Phase 1 renderer → images + coco_annotations.json (+ keypoints)
│  ├─ train.py                      # Phase 2 training wrapper around Ultralytics
│  ├─ predict.py                    # Image inference
│  ├─ video.py                      # Video inference (OpenCV)
│  ├─ refine_unsupervised.py        # Phase 3 UDA pipeline (frames→pseudo labels→YAML→fine-tune)
│  └─ helpers/
│     ├─ uda_extract_frames.py
│     ├─ uda_pseudo_label_pose.py
│     ├─ uda_temporal_filter.py
│     └─ uda_make_yaml.py
├─ synth_v8m_960/                   # Phase 2 run dir (YOLOv8m @ 960) — checkpoints/results
├─ synth_v8n_896/                   # Phase 2 run dir (YOLOv8n @ 896) - results
├─ synth_v8s_896/                   # Phase 2 run dir (YOLOv8s @ 896) - results
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt

```
## Reproduce the Results (Step-by-Step)

Below are the exact commands ( `Main.ipynb` in src) to regenerate the dataset, train the models, refine with pseudo-labels, and run inference.

All paths are given as absolute examples; feel free to switch to **`$PROJ_OUT`** for outputs.

### Phase 1 — Synthetic Data Generation

Render **1150** images with masks/boxes + **5 keypoints** per instance (`needle_holder`, `tweezers`):

```bash
OUT=/home/student/surg-pose-sim2real/outputs/synthetic_images_data

MPLBACKEND=Agg python -m blenderproc run \
  /home/student/surg-pose-sim2real/src/synthetic_data_generator.py \
  --outdir "$OUT" \
  --max_images 1150 --per_obj_cap 50 \
  --mix 'studio=0.1','hdri=0.38868677385283423','composite=0.5113132261471658' \
  --samples 24 --res_scale 0.8 --device gpu \
  --kps_dir /home/student/surg-pose-sim2real/assets/kps
```

### Optionally compute stats and sample visualizations

```bash
# Stats (images, class counts, bbox areas, visibility ratios)
python /home/student/surg-pose-sim2real/docs/phase1_stats.py \
  --outdir /home/student/surg-pose-sim2real/outputs/phase1_stats \
  --data   /home/student/surg-pose-sim2real/outputs/synthetic_images_data/coco_annotations.json

# 20 random overlays (uses original images as-is; no new backgrounds)
python /home/student/surg-pose-sim2real/docs/phase1_vis.py \
  /home/student/surg-pose-sim2real/outputs/synthetic_images_data/coco_annotations.json \
  /home/student/surg-pose-sim2real/outputs/phase1_examples_simple 20
```

### Phase 2 — Train on Synthetic Only

We trained three **YOLOv8-pose** variants using the COCO-with-keypoints dataset (via our data YAML).

```bash
DATA_YAML="/home/student/surg-pose-sim2real/outputs/datasets/surg.yaml"   # includes names, flip_idx, splits

# (1) v8n (baseline)
python -m src.train --model yolov8n-pose.pt --data $DATA_YAML \
  --imgsz 896 --epochs 150 --batch 32 \
  --project runs_pose --name synth_v8n_896

# (2) v8s (primary)
python -m src.train --model yolov8s-pose.pt --data $DATA_YAML \
  --imgsz 896 --epochs 200 --batch 24 --optimizer AdamW --lr0 0.001 --lrf 0.01 \
  --project runs_pose --name synth_v8s_896

# (3) v8m (highest accuracy)
python -m src.train --model yolov8m-pose.pt --data $DATA_YAML \
  --imgsz 960 --epochs 200 --batch 16 --optimizer AdamW --lr0 0.001 --lrf 0.01 \
  --project runs_pose --name synth_v8m_960
```

### Phase-2 results (synthetic val)

| Run           | Model            | imgsz | epochs | mAP50-95 (P) | mAP50 (P) | P (P) | R (P) |
|:--------------|:-----------------|-----:|------:|-------------:|----------:|------:|------:|
| synth_v8m_960 | yolov8m-pose.pt  |  960 |   200 |        0.752 |     0.940 | 0.949 | 0.932 |
| synth_v8s_896 | yolov8s-pose.pt  |  896 |   200 |        0.717 |     0.877 | 0.905 | 0.882 |
| synth_v8n_896 | yolov8n-pose.pt  |  896 |   150 |        0.409 |     0.839 | 0.844 | 0.848 |

Best weights are saved under each run folder (in the virual machine).  
`runs_pose/synth_v8m_960/weights/best.pt`.
### Phase 3 — Unsupervised Refinement (UDA)

We refine the **best synthetic** model using **pseudo-labels** on the unlabeled surgical video, with **OKS-NMS** per frame and a short **temporal filter** to keep only consistent tracks. Use the end-to-end script:

```bash
python /home/student/surg-pose-sim2real/src/refine_unsupervised.py \
  --teacher        /home/student/surg-pose-sim2real/runs_pose/synth_v8m_960/weights/best.pt \
  --video          /datashare/project/vids_test/4_2_24_A_1.mp4 \
  --synthetic_yaml /home/student/surg-pose-sim2real/outputs/datasets/surg.yaml \
  --imgsz 960 \
  --out_root /home/student/surg-pose-sim2real/outputs/pseudo_v1 \
  --stride 2 --det_conf 0.60 --kp_conf 0.50 --min_kps 4 \
  --train_epochs 25 --batch 16 --lr0 0.001 --device 0 --run_name refine_v1_yolov8m_960
```

**This performs:**

- **Frame extraction** (JPEG)
- **Pseudo-labeling** (keep `det-conf ≥ 0.60` and `≥ 4/5` keypoints with `kp-conf ≥ 0.50`; **OKS-NMS = 0.85**)
- **Temporal filter** (tracks `≥ 3`; match if `IoU ≥ 0.5` or `OKS ≥ 0.6`)
- **Combine YAML** (synthetic + pseudo-real)
- **Short fine-tune** (25 epochs @ 960 px, **AdamW**, cosine LR, `freeze=10`)

**Refined weights will be saved under:**  
`runs_pose/refine_v1_yolov8m_960/weights/best.pt` *(folder name depends on `--run_name`)*

**Before vs After (synthetic proxy metrics):**

- Pose **mAP50-95**: `0.752 → 0.798` *(+6.09% rel)*
- Pose **mAP50**: `0.940 → 0.951` *(+1.15% rel)*
- Box **mAP50-95**: `0.963 → 0.966` *(≈ stable)*

Qualitatively on the real video, the refined model yields **steadier tip keypoints** under glare/occlusion and **fewer misses** in clutter.


### Inference (images & video)

**Single image:**
```bash
python /home/student/surg-pose-sim2real/src/predict.py \
  --weights /home/student/surg-pose-sim2real/runs_pose/synth_v8m_960/weights/best.pt \
  --source  path/to/image.jpg --imgsz 960 --conf 0.25 --device 0
```

**Real video (synthetic-only model):**
```bash
python /home/student/surg-pose-sim2real/src/video.py \
  --source  /datashare/project/vids_test/4_2_24_A_1.mp4 \
  --out     /home/student/surg-pose-sim2real/outputs/results_synth_only.mp4 \
  --weights /home/student/surg-pose-sim2real/runs_pose/synth_v8m_960/weights/best.pt \
  --imgsz 960 --conf 0.25 --iou 0.6 --device 0 --stride 2
```
**Real video (refined model):**
```bash
python /home/student/surg-pose-sim2real/src/video.py \
  --source  /datashare/project/vids_test/4_2_24_A_1.mp4 \
  --out     /home/student/surg-pose-sim2real/outputs/results_refined.mp4 \
  --weights /home/student/surg-pose-sim2real/runs_pose/refine_v1_yolov8m_960/weights/best.pt \
  --imgsz 960 --conf 0.25 --iou 0.5 --device 0 --stride 2
```

## Pretrained Weights (Downloads)


- **Phase 2 (synthetic-only, best model):**  
  **Local:** `runs_pose/synth_v8m_960/weights/best.pt`  
  **Download:** **[phase2_synth_v8m_best](https://drive.google.com/file/d/17nBcbHIDgAFkW4it57rJFGwdQoSP90Gj/view?usp=drive_link)**

- **Phase 3 (refined, best):**  
  **Local:** `runs_pose/refine_v1_yolov8m_960/weights/best.pt`  
  **Download:** **[phase3_refined_v8m_best](https://drive.google.com/file/d/1G6L4rlptT-n0z5keilYoCGu4PpjEempq/view?usp=sharing)**

## Troubleshooting & Tips

- **BlenderProc runner not found?**  
  Ensure you can run `python -m blenderproc run -h`.  
  If local, install **Blender + BlenderProc** following their docs; on the course VM it’s pre-installed.

- **Out of GPU memory (OOM) during training?**  
  Lower `--imgsz` or `--batch` (e.g., `--imgsz 768`, `--batch 8`).

- **Flip indexing for keypoints:**  
  Make sure your `surg.yaml` includes `flip_idx` so `jaw_L`/`jaw_R` swap on horizontal flips (if you enable flips).

- **Paths:**  
  Commands above use absolute paths from the notebook; feel free to replace with `$PROJ_OUT` and `$PROJ_DATA`.



