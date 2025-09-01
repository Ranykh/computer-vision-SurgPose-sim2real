    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, yaml
from pathlib import Path

def make_yaml(syn_root, pseudo_root, out_yaml, names_csv, kpts):
    syn = Path(syn_root); pse = Path(pseudo_root)
    d = {
        "path": str(Path(out_yaml).parent),
        "train": [str(syn/"images/train"), str(pse/"images/train")],
        "val":   [str(syn/"images/val"),   str(pse/"images/val")] if (pse/"images/val").exists() else str(syn/"images/val"),
        "names": [n.strip() for n in names_csv.split(",")],
        "kpt_shape": [kpts, 3]
    }
    Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    print("Wrote:", out_yaml)
    print(yaml.safe_dump(d, sort_keys=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-root", required=True)
    ap.add_argument("--pseudo-root", required=True)
    ap.add_argument("--out-yaml", required=True)
    ap.add_argument("--names", required=True, help="comma-separated, e.g. 'needle_holder,tweezers'")
    ap.add_argument("--kpts", type=int, default=8)
    args, _ = ap.parse_known_args()
    make_yaml(args.synthetic_root, args.pseudo_root, args.out_yaml, args.names, args.kpts)
