import sys
from ultralytics import YOLO
base, data, nombre, epocas = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
extra = dict(a.split("=") for a in sys.argv[5:])
extra = {k: (float(v) if v.replace('.','',1).isdigit() else v) for k, v in extra.items()}
YOLO(base).train(data=data, epochs=epocas, imgsz=640, batch=16, workers=4, device="cpu",
                 project="/home/user/work/runs", name=nombre, exist_ok=True, patience=100,
                 plots=False, amp=False, **{"lr0": 0.002, "degrees": 10, "optimizer": "AdamW", **extra})
