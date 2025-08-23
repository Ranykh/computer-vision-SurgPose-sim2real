"""
Phase 1: Synthetic data generation
- Requirement (PDF): Generate >=1000 images and extract GT 2D pose info.
- Deliverable: synthetic_data_generator.py included in Git.
"""
import os, argparse, pathlib, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output dir for rendered images + GT")
    ap.add_argument("--count", type=int, default=1200, help="Number of images to render (>=1000 per PDF)")
    ap.add_argument("--save-gt", action="store_true", help="Save ground-truth 2D pose annotations")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    # TODO: integrate Blender/renderer; stub writes placeholders
    for i in range(args.count):
        (out / "images" / f"img_{i:05d}.png").write_bytes(b"")
        if args.save_gt:
            ann = {"file": f"img_{i:05d}.png", "keypoints": [], "visibility": []}
            (out / "labels" / f"img_{i:05d}.json").write_text(json.dumps(ann))

    (out / "manifest.json").write_text(json.dumps({"count": args.count}))
    print(f"[done] Synthetic dataset stub: {out}")

if __name__ == "__main__":
    main()
