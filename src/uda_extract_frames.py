    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2, argparse
from pathlib import Path

def extract(videos_dir, out_images, fps=5, ext="jpg"):
    out = Path(out_images); out.mkdir(parents=True, exist_ok=True)
    vids = sorted([p for p in Path(videos_dir).iterdir() if p.suffix.lower() in (".mp4",".mov",".avi",".mkv")])
    assert vids, f"No videos in {videos_dir}"
    for vi, v in enumerate(vids):
        cap = cv2.VideoCapture(str(v))
        assert cap.isOpened(), f"cannot open {v}"
        in_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        stride = max(1, int(round(in_fps / fps)))
        i = 0; saved = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if i % stride == 0:
                name = f"vid{vi:02d}_f{i:06d}.{ext}"
                cv2.imwrite(str(out / name), frame)
                saved += 1
            i += 1
        cap.release()
        print(f"{v.name}: saved {saved} frames to {out}")
    print("done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out-images", required=True)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--ext", type=str, default="jpg")
    args, _ = ap.parse_known_args()
    extract(args.videos, args.out_images, args.fps, args.ext)
  
