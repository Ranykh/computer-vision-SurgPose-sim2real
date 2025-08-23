"""
Predict on a video using OpenCV (required by PDF).
"""
import argparse, os, pathlib, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path/URL to model weights")
    ap.add_argument("--video", type=str, required=True, help="Path to input video")
    ap.add_argument("--out", type=str, default="outputs/results.mp4")
    args = ap.parse_args()
    pathlib.Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    while True:
        ok, frame = cap.read()
        if not ok: break
        # TODO: draw predictions; currently pass-through
        out.write(frame)
    out.release(); cap.release()
    print(f"[done] Wrote video (stub) to {args.out}")

if __name__ == "__main__":
    main()
