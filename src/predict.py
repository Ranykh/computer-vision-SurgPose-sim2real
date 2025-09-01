

# Simple image predictor for YOLOv8 pose.

# Usage:
#   python -m src.predict --source path/to/img.png
#   python -m src.predict --source path/to/folder --weights runs_pose/synth_v8m_960/weights/best.pt --imgsz 960


from pathlib import Path
import argparse
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str,
                    default="/home/student/surg-pose-sim2real/runs_pose/synth_v8m_960/weights/best.pt",
                    help="Path to YOLOv8-pose .pt weights")
    ap.add_argument("--source", type=str, required=True,
                    help="Image file or directory of images")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="NMS IoU threshold")
    ap.add_argument("--device", default=0, help="CUDA device id (e.g. 0) or 'cpu'")
    ap.add_argument("--name", type=str, default=None, help="Run name under runs_pose/")
    return ap.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    run_name = args.name or f"predict_{Path(args.source).stem}"
    preds = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project="runs_pose",
        name=run_name
    )

    # `preds` is a list of results objects; they share the same save_dir
    save_dir = Path(preds[0].save_dir) if preds else Path("runs_pose")/run_name
    print(f"[predict] Saved annotated outputs to: {save_dir}")


if __name__ == "__main__":
    main()


