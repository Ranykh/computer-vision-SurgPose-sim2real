# Synthetic-to-Real Pose Estimation for Surgical Tools

A reproducible pipeline for **2D pose estimation of surgical instruments** trained on **synthetic data** and refined on **unlabeled surgical video** (unsupervised domain adaptation).

> This repository implements the course/project requirements end-to-end:
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
- [License](#license)

---

## Environment Setup

**Tested:** Ubuntu 20.04/22.04 · Python 3.10+ · CUDA (any version compatible with your PyTorch build)  
**Rendering:** [BlenderProc](https://github.com/DLR-RM/BlenderProc) (invoked with `python -m blenderproc run ...`)

```bash
# 1) Clone and enter
git clone <YOUR_REPO_URL>.git
cd surg-pose-sim2real

# 2) Virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3) Install dependencies
pip install -r requirements.txt
