    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, math, numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# --------- כללי ---------
def clamp01(x): return max(0.0, min(1.0, x))

def oks(kp1_xy, kp2_xy, area, sigmas):
    #OKS בין שתי רשימות נקודות (Kx2 בפיקסלים). מתעלם מנק׳ לא־חוקיות (nan).
    kp1 = np.array(kp1_xy, dtype=np.float32); kp2 = np.array(kp2_xy, dtype=np.float32)
    vis = ~(np.isnan(kp1).any(axis=1) | np.isnan(kp2).any(axis=1))
    if vis.sum() == 0: return 0.0
    d2 = ((kp1[vis]-kp2[vis])**2).sum(axis=1)
    s2 = (sigmas[vis]*2.0)**2
    denom = s2 * (area + 1e-8)
    e = np.exp(-d2 / denom)
    return float(e.mean())

def oks_nms(preds, oks_thr=0.85, sigmas=None):
    #NMS לפי OKS בתוך פריים: preds = [{score, box_xywh (pixels), kps_xy (pixels Kx2), ...}]
    if len(preds) <= 1: return preds
    if sigmas is None: sigmas = np.full((preds[0]["kps_xy"].shape[0],), 0.05, np.float32)
    keep = []
    preds = sorted(preds, key=lambda d: d["score"], reverse=True)
    for p in preds:
        ok = True
        for q in keep:
            area = max(p["box_xywh"][2]*p["box_xywh"][3], q["box_xywh"][2]*q["box_xywh"][3])
            if oks(p["kps_xy"], q["kps_xy"], area, sigmas) >= oks_thr:
                ok = False; break
        if ok: keep.append(p)
    return keep

def format_yolo_line(cls, cx, cy, w, h, kps_xyn, kps_conf=None, kp_conf_thr=0.50):
    vals = [int(cls), f"{clamp01(cx):.6f}", f"{clamp01(cy):.6f}", f"{clamp01(w):.6f}", f"{clamp01(h):.6f}"]
    if kps_xyn is not None:
        for i in range(kps_xyn.shape[0]):
            x, y = clamp01(kps_xyn[i,0]), clamp01(kps_xyn[i,1])
            v = 2
            if kps_conf is not None:
                v = 2 if float(kps_conf[i]) >= kp_conf_thr else 1
            if x<=0 or x>=1 or y<=0 or y>=1: v = 0
            vals += [f"{x:.6f}", f"{y:.6f}", str(v)]
    return " ".join(map(str, vals))

def save_label(label_path, lines):
    p = Path(label_path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

# --------- עיקר: ריצה על תמונות ---------
def run_pseudolabels(
    images_dir, labels_dir, model_path, imgsz=640,
    conf_thr=0.50, kp_conf_thr=0.50, oks_nms_thr=0.85,
    device=None, kpts=8):
    model = YOLO(model_path)
    im_paths = sorted([p for p in Path(images_dir).iterdir()
                       if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")])
    assert im_paths, f"No images in {images_dir}"
    print(f"Found {len(im_paths)} images.")

    for i, imp in enumerate(im_paths, 1):
        im = Image.open(imp); W, H = im.size
        res = model.predict(str(imp), imgsz=imgsz, conf=0.001, iou=0.6, verbose=False, device=device)
        r = res[0]
        lines = []
        preds = []

        if r.boxes is not None and len(r.boxes) > 0:
            # boxes: normalized xywh & conf & cls
            xywhn = r.boxes.xywhn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)
            # keypoints: normalized & pixel
            kps_n = r.keypoints.xyn.cpu().numpy() if r.keypoints is not None else None
            kps_p = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
            kps_c = r.keypoints.conf.cpu().numpy() if (r.keypoints is not None and r.keypoints.conf is not None) else None

            for j in range(len(xywhn)):
                score = float(confs[j])
                if score < conf_thr:  # סף ביטחון ראשוני
                    continue
                cx, cy, w, h = xywhn[j]
                cls = int(clses[j])
                # הכנה ל-OKS-NMS
                if kps_n is not None and kps_n.shape[1] >= kpts:
                    kp_n = kps_n[j][:kpts, :2]
                    kp_p = kps_p[j][:kpts, :2]
                    kp_c = kps_c[j][:kpts] if kps_c is not None else None
                else:
                    kp_n = kp_p = kp_c = None
                preds.append({
                    "score": float(score if kp_c is None else (score * float(np.mean(kp_c)))),
                    "cls": cls,
                    "box_xywh": np.array([cx*W - (w*W)/2, cy*H - (h*H)/2, w*W, h*H], dtype=np.float32),
                    "box_n": (cx, cy, w, h),
                    "kps_xy": (np.full((kpts,2), np.nan, np.float32) if kp_p is None else kp_p),
                    "kps_xyn": (None if kp_n is None else kp_n),
                    "kps_conf": (None if kp_c is None else kp_c)
                })

            # NMS לפי OKS (מונע כפילויות כשה-NMS הבנוי לא מספיק לפוז)
            preds = oks_nms(preds, oks_thr=oks_nms_thr, sigmas=np.full((kpts,), 0.05, np.float32))

            # כתיבת שורות YOLO
            for p in preds:
                cx, cy, w, h = p["box_n"]
                ln = format_yolo_line(p["cls"], cx, cy, w, h, p["kps_xyn"], p["kps_conf"], kp_conf_thr)
                lines.append(ln)

        # גם אם אין דיטקציות → לשמור קובץ ריק (מסמן background)
        out_label = Path(labels_dir) / f"{imp.stem}.txt"
        save_label(out_label, lines)

        if i % 50 == 0:
            print(f"{i}/{len(im_paths)} frames processed")

    print("pseudo-labels saved to:", labels_dir)

if __name__ == "__main__":
    import numpy as np
    ap = argparse.ArgumentParser(description="Pseudo-label YOLOv8-Pose on unlabeled images.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels-out", required=True)
    ap.add_argument("--model", required=True, help="teacher weights (.pt) of YOLOv8-pose")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--kp-conf", type=float, default=0.50)
    ap.add_argument("--oks-nms", type=float, default=0.85)
    ap.add_argument("--device", type=str, default=None)  # e.g., '0'
    ap.add_argument("--kpts", type=int, default=8)
    args, _ = ap.parse_known_args()
    Path(args.labels_out).mkdir(parents=True, exist_ok=True)
    run_pseudolabels(args.images, args.labels_out, args.model, args.imgsz, args.conf, args.kp_conf, args.oks_nms, args.device, args.kpts)  
