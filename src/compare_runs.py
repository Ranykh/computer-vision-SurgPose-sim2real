
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import math
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --------- utils ---------
METRIC_KEYS = [
    # Pose metrics
    "metrics/mAP50-95(P)", "metrics/mAP50(P)", "metrics/precision(P)", "metrics/recall(P)",
    # Box metrics
    "metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"
]

def read_results_csv(run_dir: Path):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results.csv in {run_dir}")
    df = pd.read_csv(csv_path)
    # last epoch row:
    last = df.tail(1).iloc[0].to_dict()
    # keep entire df for curves
    return last, df

def safe_get(d, k, default=float("nan")):
    return float(d[k]) if k in d else default

def percent(delta):
    return f"{(100.0*delta):+.1f}%"

def compare_tables(base_last: dict, ref_last: dict):
    rows = []
    for k in METRIC_KEYS:
        b = safe_get(base_last, k)
        r = safe_get(ref_last, k)
        d = (r - b) if (not math.isnan(b) and not math.isnan(r)) else float("nan")
        rows.append({"metric": k, "baseline": b, "refined": r, "delta": d})
    return pd.DataFrame(rows)

def plot_curve(df_base: pd.DataFrame, df_ref: pd.DataFrame, col: str, out_png: Path, title: str):
    plt.figure()
    if col in df_base.columns:
        plt.plot(df_base.index, df_base[col], label="baseline")
    if col in df_ref.columns:
        plt.plot(df_ref.index, df_ref[col], label="refined")
    plt.xlabel("epoch")
    plt.ylabel(col)
    plt.title(title)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=180)
    plt.close()

def draw_text(img, text, x=10, y=30):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return img

def qualitative_side_by_side(
    weights_base: Path, weights_ref: Path, video_in: Path, out_dir: Path,
    imgsz=960, conf=0.25, iou=0.5, stride=10, max_frames=120, device=0
):
    out_dir.mkdir(parents=True, exist_ok=True)
    side_dir = out_dir / "frames_side_by_side"
    side_dir.mkdir(exist_ok=True)

    m_base = YOLO(str(weights_base))
    m_ref  = YOLO(str(weights_ref))

    cap = cv2.VideoCapture(str(video_in))
    assert cap.isOpened(), f"cannot open {video_in}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # writer for video comparison
    vid_path = out_dir / "qual_compare.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, fps/max(1, stride), (W*2, H))

    idx = kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, stride) == 0:
            # predict with both models
            rb = m_base.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
            rr = m_ref.predict (source=frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
            left  = rb[0].plot()
            right = rr[0].plot()
            left  = draw_text(left,  "Baseline")
            right = draw_text(right, "Refined")
            side = cv2.hconcat([left, right])

            # save jpg and write to video
            cv2.imwrite(str(side_dir / f"{idx:06d}.jpg"), side, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            writer.write(side)
            kept += 1
            if max_frames and kept >= max_frames:
                break
        idx += 1

    writer.release()
    cap.release()
    return vid_path

# --------- main ---------
def main():
    ap = argparse.ArgumentParser(description="Compare YOLOv8-pose runs: baseline vs refined")
    ap.add_argument("--base", required=True, help="Baseline run dir (contains results.csv, weights/best.pt)")
    ap.add_argument("--ref",  required=True, help="Refined run dir (contains results.csv, weights/best.pt)")
    ap.add_argument("--out",  required=True, help="Output directory for comparison artifacts")
    # optional qualitative video
    ap.add_argument("--video", type=str, default=None, help="Video path for qualitative side-by-side")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf",  type=float, default=0.25)
    ap.add_argument("--iou",   type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    base_dir = Path(args.base)
    ref_dir  = Path(args.ref)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load metrics
    base_last, base_df = read_results_csv(base_dir)
    ref_last,  ref_df  = read_results_csv(ref_dir)

    # 2) table
    table = compare_tables(base_last, ref_last)
    # pretty percent for deltas
    table["delta_%"] = table["delta"] / table["baseline"]
    table = table[["metric", "baseline", "refined", "delta", "delta_%"]]
    table.to_csv(out_dir / "comparison_table.csv", index=False)

    # quick JSON summary for the report
    summary = {}
    for _, r in table.iterrows():
        summary[r["metric"]] = {
            "baseline": float(r["baseline"]),
            "refined":  float(r["refined"]),
            "delta":    float(r["delta"]),
            "delta_pct": float(r["delta_%"]) if pd.notna(r["delta_%"]) else None
        }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))

    # 3) curves
    base_df = base_df.reset_index(drop=True)
    ref_df  = ref_df.reset_index(drop=True)
    plot_curve(base_df, ref_df, "metrics/mAP50-95(P)", out_dir / "curve_mAP5095_pose.png", "Pose mAP50-95 per epoch")
    plot_curve(base_df, ref_df, "metrics/mAP50(P)",    out_dir / "curve_mAP50_pose.png",    "Pose mAP50 per epoch")
    plot_curve(base_df, ref_df, "metrics/mAP50-95(B)", out_dir / "curve_mAP5095_box.png",   "Box mAP50-95 per epoch")
    plot_curve(base_df, ref_df, "metrics/mAP50(B)",    out_dir / "curve_mAP50_box.png",     "Box mAP50 per epoch")

    # 4) qualitative side-by-side (optional)
    if args.video:
        vid = qualitative_side_by_side(
            weights_base=base_dir/"weights/best.pt",
            weights_ref= ref_dir/"weights/best.pt",
            video_in=Path(args.video),
            out_dir=out_dir/"qualitative",
            imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            stride=args.stride, max_frames=args.max_frames, device=args.device
        )
        print(f"[compare] Wrote qualitative video: {vid}")

    print(f"[compare] Done. Artifacts in: {out_dir}")

if __name__ == "__main__":
    main()


