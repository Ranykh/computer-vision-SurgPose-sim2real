


# src/train.py
import argparse, os, json
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-pose.pt", help="Ultralytics model name or path")
    ap.add_argument("--data",  default="data/surg_pose.yaml", help="YAML with paths, kpt_shape, flip_idx")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs_pose")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=0.001)   # a bit lower for pose stability
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--multi_scale", action="store_true")
    ap.add_argument("--close_mosaic", type=int, default=15)  # last N epochs without mosaic
    return ap.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        patience=args.patience,
        multi_scale=args.multi_scale,
        close_mosaic=args.close_mosaic,
        cache=True,
        pretrained=True,
        plots=True,
        val=True,
        save=True,
        exist_ok=True
    )

    if args.resume:
        results = model.train(resume=True, **train_kwargs)
    else:
        results = model.train(**train_kwargs)

    # Validate on val set and print key metrics
    val_res = model.val(project=args.project, name=f"{args.name}_val", plots=True)
    print("Best weights:", model.ckpt_path)
    print("Val metrics:", val_res)

if __name__ == "__main__":
    main()

