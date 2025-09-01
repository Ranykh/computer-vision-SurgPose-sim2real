


# Video predictor for YOLOv8 pose (OpenCV pipeline).

# Usage:
#   python -m src.video --source /path/in.mp4 --out outputs/results_synthetic_only.mp4 #       --weights runs_pose/synth_v8m_960/weights/best.pt --imgsz 960 --conf 0.25


from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str,
                    default="/home/student/surg-pose-sim2real/runs_pose/synth_v8m_960/weights/best.pt",
                    help="Path to YOLOv8-pose .pt weights")
    ap.add_argument("--source", type=str, required=True,
                    help="Input video path")
    ap.add_argument("--out", type=str, default="/home/student/surg-pose-sim2real/outputs/results_synthetic_only.mp4",
                    help="Output video path")
    ap.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--device", default=0, help="CUDA device id (e.g. 0) or 'cpu'")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--show", action="store_true", help="Optional: display live preview window")
    return ap.parse_args()


def make_writer(cap, out_path: Path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely compatible
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))


def main():
    args = parse_args()
    src = Path(args.source)
    out_path = Path(args.out)
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {src}")

    writer = make_writer(cap, out_path)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % max(1, args.stride) == 0:
            # run inference on this frame
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False
            )
            # results is a list with one element for a single frame input
            plotted = results[0].plot()  # draw boxes + keypoints
        else:
            plotted = frame  # skip processing, keep original frame

        writer.write(plotted)

        if args.show:
            cv2.imshow("pred", plotted)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"[video] Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()

