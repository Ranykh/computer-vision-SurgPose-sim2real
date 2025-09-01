    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, numpy as np, math, re
from pathlib import Path
from PIL import Image

def clamp01(x): return max(0.0, min(1.0, x))
def iou_xywhn(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax1, ay1 = ax - aw/2, ay - ah/2
    ax2, ay2 = ax + aw/2, ay + ah/2
    bx1, by1 = bx - bw/2, by - bh/2
    bx2, by2 = bx + bw/2, by + bh/2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw*ah + bw*bh - inter + 1e-8
    return inter / union

def oks(kp1, kp2, area, sigmas):
    kp1 = np.array(kp1); kp2 = np.array(kp2)
    vis = ~np.isnan(kp1).any(axis=1) & ~np.isnan(kp2).any(axis=1)
    if vis.sum()==0: return 0.0
    d2 = ((kp1[vis]-kp2[vis])**2).sum(axis=1)
    e = np.exp(-d2 / ((sigmas[vis]*2.0)**2 * (area + 1e-8)))
    return float(e.mean())

def parse_label_line(s, kpts=8):
    p = [float(x) for x in s.strip().split()]
    if len(p) < 5: return None
    cls = int(p[0]); cx,cy,w,h = p[1:5]
    kps = np.full((kpts,2), np.nan, np.float32)
    if len(p) >= 5+3*kpts:
        arr = p[5:5+3*kpts]
        for i in range(kpts):
            x, y, v = arr[3*i:3*i+3]
            if v > 0:
                kps[i,0], kps[i,1] = x, y
    return {"cls":cls, "box":(cx,cy,w,h), "kps":kps}

def load_labels(lbl_path, kpts=8):
    if not lbl_path.exists(): return []
    lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]
    items = [parse_label_line(ln, kpts) for ln in lines]
    return [x for x in items if x is not None]

def save_labels(lbl_path, items, kpts=8):
    out = []
    for it in items:
        cls = it["cls"]; cx,cy,w,h = it["box"]; kps = it["kps"]
        vals = [str(cls), f"{cx:.6f}",f"{cy:.6f}",f"{w:.6f}",f"{h:.6f}"]
        if not np.isnan(kps).all():
            for i in range(kpts):
                x, y = kps[i]
                v = 2 if (0<x<1 and 0<y<1 and not np.isnan(x) and not np.isnan(y)) else 0
                vals += [f"{clamp01(x):.6f}", f"{clamp01(y):.6f}", str(v)]
        out.append(" ".join(vals))
    lbl_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")

def sort_key_natural(p: Path):
    t = re.split(r'(\d+)', p.stem)
    return [int(s) if s.isdigit() else s for s in t]

def run(images_dir, labels_dir, min_track=3, iou_thr=0.5, oks_thr=0.6, kpts=8):
    sigmas = np.full((kpts,), 0.05, np.float32)
    ims = sorted([p for p in Path(images_dir).iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")],
                 key=sort_key_natural)
    prev = []  # [{id, item}]
    next_id = 1
    tracks = {}  # id -> list[(frame_index, item_index)]
    all_items = []  # per-frame items

    for fi, imp in enumerate(ims):
        items = load_labels((Path(labels_dir)/ f"{imp.stem}.txt"), kpts)
        all_items.append(items)
        # התאמה לפריים קודם (גרידיות)
        assigned = set()
        new_prev = []
        for pi, p in enumerate(items):
            best = None; best_score = -1
            for tid, prev_item in prev:
                # ציון משולב IoU/OKS
                iou = iou_xywhn(p["box"], prev_item["box"])
                area = max(p["box"][2]*p["box"][3], prev_item["box"][2]*prev_item["box"][3])
                oksv = oks(p["kps"], prev_item["kps"], area, sigmas)
                score = 0.5*iou + 0.5*oksv
                if (iou >= iou_thr or oksv >= oks_thr) and score > best_score:
                    best = tid; best_score = score
            if best is not None:
                tracks.setdefault(best, []).append((fi, pi))
                new_prev.append((best, p))
                assigned.add(pi)

        # לא משויכים → פתיחת מסלולים
        for pi, p in enumerate(items):
            if pi in assigned: continue
            tracks[next_id] = [(fi, pi)]
            new_prev.append((next_id, p))
            next_id += 1

        prev = new_prev

    # מחיקה של מסלולים קצרים
    keep_mask_per_frame = [np.zeros(len(items), dtype=bool) for items in all_items]
    for tid, lst in tracks.items():
        if len(lst) >= min_track:
            for fi, pi in lst:
                keep_mask_per_frame[fi][pi] = True

    # כתיבה חזרה (מסירים פריטים שלא שרדו טמפורלית)
    ims2 = list(ims)
    for fi, imp in enumerate(ims2):
        items = [it for it, keep in zip(all_items[fi], keep_mask_per_frame[fi]) if keep]
        save_labels((Path(labels_dir)/ f"{imp.stem}.txt"), items, kpts)
    print("Temporal filter done. kept detections with tracks ≥", min_track)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Temporal consistency filter for pseudo-labeled YOLOv8-pose.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--min-track", type=int, default=3)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--oks-thr", type=float, default=0.6)
    ap.add_argument("--kpts", type=int, default=8)
    args, _ = ap.parse_known_args()
    run(args.images, args.labels, args.min_track, args.iou_thr, args.oks_thr, args.kpts)           
