"""
Phase 1: Synthetic data generation
- Requirement (PDF): Generate >=1000 images and extract GT 2D pose info.
- Deliverable: synthetic_data_generator.py included in Git.
"""
# Wrapper that runs BlenderProc with src/bp_render.py
import argparse, os, subprocess, sys, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir for renders + COCO")
    ap.add_argument("--count", type=int, default=100, help="Total images to generate (>=1000 for the project)")
    ap.add_argument("--per_obj_cap", type=int, default=100, help="Max frames per single object")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure PROJ_DATA is set (defaults to /datashare/project)
    os.environ.setdefault("PROJ_DATA", "/datashare/project")

    cmd = [
        sys.executable, "-m", "blenderproc", "run",
        "src/bp_render.py",
        "--outdir", str(outdir),
        "--max_images", str(args.count),
        "--per_obj_cap", str(args.per_obj_cap),
    ]
    print("[synthetic_data_generator] Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[synthetic_data_generator] Done.")

if __name__ == "__main__":
    main()
