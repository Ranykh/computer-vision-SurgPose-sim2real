

# /home/student/surg-pose-sim2real/src/refine_unsupervised.py
from __future__ import annotations
####################
import argparse
from pathlib import Path
import cv2
import yaml
import numpy as np
from ultralytics import YOLO


def dprint(*a):
    print("[refine]", *a)


# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Unsupervised refinement via pseudo-labeling for YOLOv8-pose")
    # Inputs
    ap.add_argument("--teacher", type=str, required=True, help="Path to synthetic-only best.pt")
    ap.add_argument("--video", type=str, required=True, help="Unlabeled video path")
    ap.add_argument("--synthetic_yaml", type=str, required=True, help="Phase-1/2 dataset YAML")
    ap.add_argument("--imgsz", type=int, default=960)

    # Pseudo-labeling controls
    ap.add_argument("--out_root", type=str, required=True, help="Where to write pseudo dataset")
    ap.add_argument("--stride", type=int, default=2, help="Sample every Nth frame")
    ap.add_argument("--det_conf", type=float, default=0.60)
    ap.add_argument("--kp_conf", type=float, default=0.50)
    ap.add_argument("--min_kps", type=int, default=4)  # out of K
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")

    # Fine-tune controls
    ap.add_argument("--train_epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--device", default=0)
    ap.add_argument("--run_name", type=str, default="refine_v1")
    return ap.parse_args()


# ---------- io helpers ----------
def ensure_dirs(root: Path):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def write_yolo_label_file(path: Path, lines: list[str]):
    #Write one YOLO label file per image.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if lines:
            for line in lines:
                f.write(line + "\n")


# ---------- pseudo-label generation ----------
def extract_frames(video_path: Path, dst_images: Path, stride: int,
                   max_frames: int = 0, ext: str = "jpg"):
   #Save frames as JPG to avoid libpng spam and keep disk use lower.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
    idx, kept = 0, 0
    dst_images.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, stride) == 0:
            name = f"{idx:06d}.{ext.lower()}"
            fp = dst_images / name
            cv2.imwrite(str(fp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            kept += 1
            if max_frames and kept >= max_frames:
                break
        idx += 1
    cap.release()
    dprint(f"extracted {kept} frames -> {dst_images}")
    return kept


def generate_pseudo_labels(model: YOLO, frames_dir: Path, labels_dir: Path,
                           det_thr: float, kp_thr: float, min_kps: int, imgsz: int):
    images = (
        sorted(frames_dir.glob("*.jpg"))
        + sorted(frames_dir.glob("*.jpeg"))
        + sorted(frames_dir.glob("*.png"))
    )
    kept_images = 0
    labels_dir.mkdir(parents=True, exist_ok=True)

    for chunk_start in range(0, len(images), 64):
        batch = [str(p) for p in images[chunk_start:chunk_start + 64]]
        if not batch:
            break

        results = model.predict(
            source=batch, imgsz=imgsz, conf=det_thr, iou=0.5,
            verbose=False, save=False, device=model.device
        )

        for res, img_path in zip(results, batch):
            lines: list[str] = []

            if res.boxes is not None and len(res.boxes) > 0 and res.keypoints is not None:
                boxes = res.boxes
                kpts = res.keypoints
                xywhn_all = boxes.xywhn.cpu().numpy()
                cls_all   = boxes.cls.cpu().numpy().astype(int)
                conf_all  = boxes.conf.cpu().numpy()
                kxy_all   = kpts.xyn.cpu().numpy()
                kconf_all = (kpts.conf.cpu().numpy() if kpts.conf is not None else None)

                for i in range(xywhn_all.shape[0]):
                    if conf_all[i] < det_thr:
                        continue
                    kxy = kxy_all[i]
                    kcf = kconf_all[i] if kconf_all is not None else None
                    vis_flags = (kcf >= kp_thr).sum() if kcf is not None else kxy.shape[0]
                    if vis_flags < min_kps:
                        continue

                    cx, cy, w, h = xywhn_all[i].tolist()
                    cls_i = int(cls_all[i])

                    trips: list[str] = []
                    for kk in range(kxy.shape[0]):
                        x = float(kxy[kk, 0])
                        y = float(kxy[kk, 1])
                        c = float(kcf[kk]) if kcf is not None else 1.0
                        v = 2 if c >= kp_thr else (1 if c > 0 else 0)
                        x = min(max(x, 0.0), 1.0)
                        y = min(max(y, 0.0), 1.0)
                        trips += [f"{x:.6f}", f"{y:.6f}", str(int(v))]

                    line = " ".join(
                        [str(cls_i), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"] + trips
                    )
                    lines.append(line)

            out_txt = labels_dir / (Path(img_path).stem + ".txt")
            write_yolo_label_file(out_txt, lines)
            if lines:
                kept_images += 1

    dprint(f"kept {kept_images} images with ≥1 pseudo-label (label files exist for all frames)")
    return kept_images


# ---------- YAML combo ----------
def build_combined_yaml(synth_yaml_path: Path, pseudo_root: Path, out_yaml: Path):
    sy = read_yaml(synth_yaml_path)
    train_paths = sy["train"] if isinstance(sy["train"], list) else [sy["train"]]
    train_paths = [str(Path(p)) for p in train_paths]
    train_paths.append(str(pseudo_root / "images"))

    combo = {
        "names": sy["names"],
        "kpt_shape": sy.get("kpt_shape", [5, 3]),
        "train": train_paths,
        "val": sy["val"],           # keep synthetic val as proxy
        "imgsz": sy.get("imgsz", None)
    }
    write_yaml(combo, out_yaml)
    dprint("wrote combined YAML ->", out_yaml)
    return out_yaml


# ---------- training ----------
def finetune(teacher_pt: Path, data_yaml: Path, run_name: str,
             epochs: int, batch: int, lr0: float, imgsz: int, device):
    model = YOLO(str(teacher_pt))
    results = model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        lr0=float(lr0),
        cos_lr=True,
        close_mosaic=10,
        pretrained=False,   # weights already loaded above
        project="runs_pose",
        name=run_name,
        freeze=10,          # freeze early backbone; tune if needed
        patience=20,
        cache=True
    )
    dprint("train complete; best weights at:", results.best)
    return results.best


def main():
    args = parse_args()

    teacher_pt = Path(args.teacher)
    video = Path(args.video)
    synth_yaml = Path(args.synthetic_yaml)
    pseudo_root = Path(args.out_root)
    ensure_dirs(pseudo_root)
    frames_dir = pseudo_root / "images"
    labels_dir = pseudo_root / "labels"

    # 1) extract frames
    n = extract_frames(video, frames_dir, args.stride, args.max_frames)
    if n == 0:
        raise RuntimeError("No frames extracted.")

    # 2) pseudo-labeling
    teacher = YOLO(str(teacher_pt))
    kept = generate_pseudo_labels(
        model=teacher,
        frames_dir=frames_dir,
        labels_dir=labels_dir,
        det_thr=args.det_conf,
        kp_thr=args.kp_conf,
        min_kps=args.min_kps,
        imgsz=args.imgsz
    )
    if kept == 0:
        dprint("WARNING: no pseudo-labeled frames passed thresholds.")

    # 3) combined YAML
    combo_yaml = build_combined_yaml(synth_yaml, pseudo_root, pseudo_root / "combined.yaml")

    # 4) fine-tune
    if args.train_epochs > 0:
        best = finetune(
            teacher_pt, combo_yaml, args.run_name,
            args.train_epochs, args.batch, args.lr0, args.imgsz, args.device
        )
        print(best)


if __name__ == "__main__":
    main()



