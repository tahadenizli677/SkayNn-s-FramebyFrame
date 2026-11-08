# YOLOv8 Arsenal training + ONNX export - Python 3.11+, Windows 11, ultralytics
# input: dataset/data.yaml from fetch_dataset.py
# output: best.onnx in repo root -> aimbot.exe picks it up without rebuild

import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

DATA_YAML = "dataset/data.yaml"
EPOCHS = 120
OUT = Path("runs/detect/arsenal/weights")


def main() -> None:
    if not Path(DATA_YAML).exists():
        sys.exit("run fetch_dataset.py first")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8s.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        device=device,
        optimizer="auto",
        patience=30,
        name="arsenal",
        # roblox art flips between maps -> hsv_v + scale keeps detector alive cross-map
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, close_mosaic=15,
    )

    best = OUT / "best.pt"
    if not best.exists():
        sys.exit("training produced no weights")

    metrics = YOLO(str(best)).val()
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")

    # onnx, no NMS node -> aimbot filters itself, torch not needed at runtime
    YOLO(str(best)).export(format="onnx", imgsz=640, simplify=True, opset=13)
    shutil.copy(OUT / "best.onnx", "best.onnx")
    print(f"model -> {Path.cwd() / 'best.onnx'}")


if __name__ == "__main__":
    main()
