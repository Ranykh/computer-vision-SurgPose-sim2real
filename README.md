# surg-pose-sim2real

A reproducible pipeline for **2D pose estimation of surgical instruments** trained on **synthetic data** and refined on **unlabeled surgical video** (unsupervised domain adaptation).

> This repo follows the project PDF requirements: Phase 1 (synthetic data), Phase 2 (model training + video inference), Phase 3 (unsupervised refinement), and the submission instructions (single PDF, GitHub repo with scripts, and result videos).

---

## 0) Quickstart (VM)

```bash
# (A) Clone and enter
git clone <YOUR_REPO_URL>.git
cd surg-pose-sim2real

# (B) Python env (choose Python 3.10+ locally)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# (C) Verify VM resources (expects /datashare/project per PDF)
bash scripts/verify_resources.sh

# (D) Configure paths (optional)
export PROJ_DATA=/datashare/project     # default used if unset
export PROJ_OUT=$(pwd)/outputs          # where renders/logs go
mkdir -p "$PROJ_OUT"

# (E) Dry runs
python src/synthetic_data_generator.py --help
python src/predict.py --help
python src/video.py --help
python src/refine_unsupervised.py --help
```

> **Note**: OS/Python/CUDA versions are not specified in the PDF. Choose and document them here (e.g., Ubuntu 22.04, Python 3.10, CUDA 12.x).

---

## 1) Accessing VM Resources

All provided assets are located at **`/datashare/project`** on your VM (per the PDF). The repo reads from the environment variable `PROJ_DATA` (defaults to `/datashare/project`).

Expected subfolders/files (names may vary):
- `camera.json` (camera intrinsics)
- `cad/` with `.obj` instrument models (articulations)
- `backgrounds/coco2017/` images
- `hdris/polyhaven/` HDR files
- `videos/4_2_24_A_1.mp4` (unlabeled surgical video)

Use:
```bash
bash scripts/verify_resources.sh
```

This will check presence and show counts (e.g., *.obj found).

---

## 2) Repo Layout

```
surg-pose-sim2real/
├─ src/
│  ├─ synthetic_data_generator.py   # Phase 1
│  ├─ train.py                      # Phase 2
│  ├─ predict.py                    # image inference
│  ├─ video.py                      # video inference (OpenCV)
│  └─ refine_unsupervised.py        # Phase 3 (pseudo-labels)
├─ configs/
│  ├─ default.yaml
│  ├─ model_yolov8_pose.yaml
│  └─ refinement.yaml
├─ scripts/
│  ├─ verify_resources.sh
│  ├─ setup_env.sh
│  └─ run_all.sh
├─ docs/
│  └─ report_template.md
├─ notebooks/                       # optional exploration
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

---

## 3) Typical Flow

1. **Verify assets:** `bash scripts/verify_resources.sh`  
2. **Phase 1 – Generate synthetic dataset:**  
   ```bash
   python src/synthetic_data_generator.py      --out $PROJ_OUT/synth      --count 1200 --variations pose lighting background creative2      --save-gt
   ```
3. **Phase 2 – Train on synthetic only:**  
   ```bash
   python src/train.py --data $PROJ_OUT/synth --cfg configs/model_yolov8_pose.yaml --out $PROJ_OUT/train_yolo
   python src/video.py --weights $PROJ_OUT/train_yolo/best.pt --video $PROJ_DATA/videos/4_2_24_A_1.mp4      --out $PROJ_OUT/results_synthetic_only.mp4
   ```
4. **Phase 3 – Unsupervised refinement:**  
   ```bash
   python src/refine_unsupervised.py      --weights $PROJ_OUT/train_yolo/best.pt      --video $PROJ_DATA/videos/4_2_24_A_1.mp4      --out $PROJ_OUT/refine
   python src/video.py --weights $PROJ_OUT/refine/best.pt --video $PROJ_DATA/videos/4_2_24_A_1.mp4      --out $PROJ_OUT/results_refined.mp4
   ```
5. **Prepare submission:** Put links to weights in README, and compile a single ≤6-page PDF.

---

## 4) Repro & Weights

When done training/refinement, upload weights to a storage bucket and **paste the download links here** (Phase 2 + Phase 3).

---

## 5) License

Choose a license (e.g., Apache-2.0). Ensure external models respect their own licenses.
