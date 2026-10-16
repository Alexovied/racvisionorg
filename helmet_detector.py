"""Detector de cabezas con y sin casco (YOLOv8 exportado a ONNX).

Encuentra cabezas desde cualquier ángulo (de frente, de costado, de espaldas
o con la cara tapada) y dice si tienen casco. Corre con OpenVINO (Intel) y,
si no está instalado o falla, con cv2.dnn; no hace falta PyTorch ni
Ultralytics en el servidor.

El modelo se entrena con entrenamiento/detector_casco_colab.ipynb y se copia
como detector_casco.onnx en la raíz del proyecto. Si el archivo no está,
`disponible()` devuelve False y app.py usa el esquema anterior (cara + Keras).
"""
import os
import threading
import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_casco.onnx")

INPUT_SIZE    = 640
CONF_THRESHOLD = 0.35   # mismo umbral con el que el notebook mide aciertos
NMS_THRESHOLD  = 0.45
CLASES = ("casco", "sin_casco")

try:
    import openvino
    _core = openvino.Core()
except Exception as e:  # sin OpenVINO se usa cv2.dnn
    print(f"[!] OpenVINO no disponible ({e}); los YOLO corren con cv2.dnn")
    _core = None


def cargar_modelo(path):
    """Compila el ONNX con OpenVINO en CPU; si no se puede, lo carga con cv2.dnn.
    Devuelve (red, "OpenVINO" | "cv2.dnn"). cv2.dnn lanza cv2.error si el archivo no sirve."""
    if _core is not None:
        try:
            return _core.compile_model(path, "CPU"), "OpenVINO"
        except Exception as e:
            print(f"[!] OpenVINO no pudo compilar {os.path.basename(path)} ({e}); se usa cv2.dnn")
    return cv2.dnn.readNetFromONNX(path), "cv2.dnn"


_net = None
_lock = threading.Lock()  # ni cv2.dnn.Net ni el CompiledModel de OpenVINO son seguros entre hilos
if os.path.exists(MODEL_PATH):
    try:
        _net, motor = cargar_modelo(MODEL_PATH)
        print(f"[+] Detector de cascos: YOLO (detector_casco.onnx) con {motor}")
    except cv2.error as e:
        print(f"[!] No se pudo cargar detector_casco.onnx ({e})")


def disponible():
    return _net is not None


def _letterbox(img):
    """Escala sin deformar y rellena con gris hasta INPUT_SIZE, como en el entrenamiento."""
    h, w = img.shape[:2]
    r = min(INPUT_SIZE / h, INPUT_SIZE / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    pad_x, pad_y = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, pad_x, pad_y


def correr_yolo(net, lock, img, n_clases, conf_th=CONF_THRESHOLD, por_clase=False):
    """Corre un YOLOv8 exportado a ONNX (red de cargar_modelo) y devuelve [(bbox [x, y, w, h], puntajes por clase)].
    por_clase=False: NMS sin distinguir clase (una cabeza es "casco" o "sin_casco").
    por_clase=True: cada clase aparte (una persona y su chaleco se superponen)."""
    h_img, w_img = img.shape[:2]
    canvas, r, pad_x, pad_y = _letterbox(img)
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)

    with lock:
        if hasattr(net, "setInput"):  # cv2.dnn.Net
            net.setInput(blob)
            out = net.forward()
        else:                         # openvino.CompiledModel (copia la salida)
            out = net(blob)[0]

    # YOLOv8: (1, 4 + clases, N) → (N, 4 + clases); cajas en cx, cy, w, h del canvas
    preds = out[0].T
    scores = preds[:, 4:4 + n_clases]
    conf = scores.max(axis=1)
    keep = conf >= conf_th
    preds, scores, conf = preds[keep], scores[keep], conf[keep]
    if len(preds) == 0:
        return []

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x = (cx - bw / 2 - pad_x) / r
    y = (cy - bh / 2 - pad_y) / r
    boxes = np.stack([x, y, bw / r, bh / r], axis=1)

    if por_clase:
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf.tolist(), scores.argmax(axis=1).tolist(),
                                      conf_th, NMS_THRESHOLD)
    else:
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), conf_th, NMS_THRESHOLD)
    resultados = []
    for i in np.array(idx).flatten():
        bx, by, bw_i, bh_i = boxes[i]
        x0, y0 = max(0, int(bx)), max(0, int(by))
        x1, y1 = min(w_img, int(bx + bw_i)), min(h_img, int(by + bh_i))
        if x1 > x0 and y1 > y0:
            resultados.append(([x0, y0, x1 - x0, y1 - y0], scores[i]))
    return resultados


def detectar(img):
    """Devuelve una lista de cabezas detectadas:
    {"bbox": [x, y, w, h], "clase": "casco"|"sin_casco", "conf": float,
     "p_casco": float, "p_sin": float}  (p_casco + p_sin = 1)
    """
    if _net is None or img is None:
        return []
    resultados = []
    for bbox, sc in correr_yolo(_net, _lock, img, len(CLASES)):
        s_casco, s_sin = float(sc[0]), float(sc[1])
        total = s_casco + s_sin
        resultados.append({
            "bbox": bbox,
            "clase": CLASES[int(sc.argmax())],
            "conf": float(sc.max()),
            "p_casco": s_casco / total,
            "p_sin": s_sin / total,
        })
    return sorted(resultados, key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)
