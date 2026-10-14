"""Chaleco reflectante: se busca color de alta visibilidad en el torso.

Los chalecos de seguridad usan colores fluorescentes normalizados (amarillo
lima o naranja), que casi no aparecen en ropa común ni en fondos. Debajo de
cada cabeza que encuentra el detector de cascos se mide qué parte del torso
tiene ese color.

Medido con fotos del proyecto: con chaleco 9-49 % del torso (9 % de costado),
sin chaleco 0 %, y 0-1 % en 40 fotos del dataset sin chalecos. El umbral 4 %
queda en el medio.

shortcut: regla de color, no un modelo entrenado. Una remera amarilla
fluorescente cuenta como chaleco; si eso pasa en la planta, entrenar YOLO con
fotos de chalecos (mismo notebook que el de cascos).
"""
import os
import threading

import cv2

import helmet_detector

UMBRAL = 0.04


def _torso(img, bbox):
    """Región del torso debajo de la cabeza, o None si casi no entra en la imagen."""
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x0, x1 = max(0, int(x - 0.6 * w)), min(W, int(x + 1.6 * w))
    y0, y1 = min(H, int(y + 1.1 * h)), min(H, int(y + 3.6 * h))
    # Cámara muy cerca (solo se ve la cara): no se puede juzgar el chaleco
    if y1 - y0 < 1.2 * h or x1 - x0 < w:
        return None
    return img[y0:y1, x0:x1]


def proporcion_alta_visibilidad(region):
    hh, ss, vv = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2HSV))
    amarillo = (hh >= 25) & (hh <= 45) & (ss >= 110) & (vv >= 140)
    naranja = (hh >= 4) & (hh <= 18) & (ss >= 150) & (vv >= 150)
    return float((amarillo | naranja).mean())


def tiene_chaleco(img, bbox):
    """Regla de color: True / False, o None si el torso no está a la vista."""
    torso = _torso(img, bbox)
    if torso is None:
        return None
    return proporcion_alta_visibilidad(torso) >= UMBRAL


# ── Modelo entrenado (entrenamiento/detector_chaleco_colab.ipynb) ────────────
# YOLOv8n con dos clases: 0 = persona, 1 = chaleco. Si el archivo no está, se
# usa la regla de color de arriba.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_chaleco.onnx")
CONF_MODELO = 0.35
_net, _lock = None, threading.Lock()
if os.path.exists(MODEL_PATH):
    try:
        _net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        print("[+] Detector de chalecos: YOLO (detector_chaleco.onnx)")
    except cv2.error as e:
        print(f"[!] No se pudo cargar detector_chaleco.onnx ({e}); se usa la regla de color")


def _dentro(punto, caja):
    x, y, w, h = caja
    return x <= punto[0] <= x + w and y <= punto[1] <= y + h


def chalecos_por_cabeza(img, cabezas):
    """Para cada caja de cabeza: True / False / None (torso no visible).
    Con el modelo: la persona que contiene esa cabeza tiene o no un chaleco
    adentro. Si el modelo no encuentra a la persona, se usa la regla de color."""
    if _net is None or not cabezas:
        return [tiene_chaleco(img, c) for c in cabezas]
    dets = helmet_detector.correr_yolo(_net, _lock, img, 2, CONF_MODELO, por_clase=True)
    personas = [b for b, sc in dets if sc.argmax() == 0]
    chalecos = [b for b, sc in dets if sc.argmax() == 1]
    resultado = []
    for (x, y, w, h) in cabezas:
        centro = (x + w / 2, y + h / 2)
        suyas = [p for p in personas if _dentro(centro, p) and centro[1] <= p[1] + 0.5 * p[3]]
        if not suyas:
            resultado.append(tiene_chaleco(img, [x, y, w, h]))
            continue
        p = max(suyas, key=lambda b: b[2] * b[3])
        if p[3] < 2 * h:          # de la persona casi solo se ve la cabeza
            resultado.append(None)
            continue
        # El modelo falla con chalecos de costado (7 de 10 en las fotos de Luciano) y la
        # regla de color no (10 de 10); ninguno marcó chaleco en 60 cabezas sin chaleco.
        # Se combinan: alcanza con que uno de los dos vea el chaleco.
        resultado.append(any(_dentro((c[0] + c[2] / 2, c[1] + c[3] / 2), p) for c in chalecos)
                         or bool(tiene_chaleco(img, [x, y, w, h])))
    return resultado
