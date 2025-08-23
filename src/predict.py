"""
Predict on a single image (required by PDF).
"""
import argparse, pathlib, os
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path/URL to model weights")
    ap.add_argument("--image", type=str, required=True, help="Path to input image")
    ap.add_argument("--out", type=str, default="outputs/pred.jpg")
    args = ap.parse_args()
    pathlib.Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    # TODO: implement model load & inference; stub copies the image
    Image.open(args.image).save(args.out)
    print(f"[done] Saved prediction (stub) to {args.out}")

if __name__ == "__main__":
    main()
