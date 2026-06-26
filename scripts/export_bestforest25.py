from pathlib import Path
from shutil import copy2

from ultralytics import YOLO


MODEL_PATH = "bestforest25.pt"
PUBLIC_MODEL_PATH = Path("public/bestforest25.onnx")


if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    exported = Path(
        model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False, nms=False)
    )
    PUBLIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    copy2(exported, PUBLIC_MODEL_PATH)
    if exported.resolve() != PUBLIC_MODEL_PATH.resolve():
        exported.unlink()
    print(f"ONNX siap untuk web: {PUBLIC_MODEL_PATH}")
