"""
Phase 1: Synthetic data generation
- Requirement (PDF): Generate >=1000 images and extract GT 2D pose info.
- Deliverable: synthetic_data_generator.py included in Git.
"""
# Wrapper that runs BlenderProc with src/bp_render.py
# src/synthetic_data_generator.py
import argparse, os, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir for renders + COCO")
    ap.add_argument("--count", type=int, default=120, help="Total images to generate (>=1000)")
    ap.add_argument("--per_obj_cap", type=int, default=20, help="Max frames per single object")
    args = ap.parse_args()

    # Resolve absolute paths so Blender can find the script regardless of its own CWD
    bp_script = (Path(__file__).parent / "bp_render.py").resolve()
    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure PROJ_DATA is set
    os.environ.setdefault("PROJ_DATA", "/datashare/project")


    cmd = [
        sys.executable, "-m", "blenderproc", "run",
        str(bp_script),                  # <— ABSOLUTE path
        "--outdir", str(outdir),
        "--max_images", str(args.count),
        "--per_obj_cap", str(args.per_obj_cap),
    ]
    print("[synthetic_data_generator] bp_script:", bp_script)
    print("[synthetic_data_generator] outdir:", outdir)
    subprocess.run(cmd, check=True)
    print("[synthetic_data_generator] Done.")

if __name__ == "__main__":
    main()
