"""Pose de las personas (YOLOv8n-pose exportado a ONNX): 17 puntos del cuerpo por persona.

El motor de cámaras lo usa para el hombre caído: mide la inclinación del torso
(hombros → caderas) en vez de mirar solo la forma de la caja. Si detector_pose.onnx
no está, `disponible()` devuelve False y motor.py sigue con la regla de la caja.
"""
import math
import os
import threading

import cv2

import helmet_detector

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_pose.onnx")
CONF_PERSONA = 0.35
CONF_PUNTO = 0.5          # por debajo, el punto no se ve bien y no se usa
ANGULO_CAIDO = 50         # grados del torso respecto de la vertical
HOMBROS, CADERAS = (5, 6), (11, 12)   # índices COCO

_net, _lock = None, threading.Lock()
if os.path.exists(MODEL_PATH):
    try:
        _net, motor = helmet_detector.cargar_modelo(MODEL_PATH)
        print(f"[+] Detector de pose: YOLO (detector_pose.onnx) con {motor}")
    except cv2.error as e:
        print(f"[!] No se pudo cargar detector_pose.onnx ({e}); el hombre caído usa la forma de la caja")


def disponible():
    return _net is not None


def detectar(img):
    """[(bbox [x, y, w, h], puntos (17, 3): x, y, confianza)] de cada persona."""
    if _net is None:
        return []
    return [(bbox, puntos) for bbox, _, puntos in
            helmet_detector.correr_yolo(_net, _lock, img, 1, CONF_PERSONA, puntos=True)]


def torso_caido(puntos):
    """True si el torso está acostado o cabeza abajo; False si está derecho;
    None si no se ven bien los hombros y las caderas (muy cerca de la cámara, tapado)."""
    if min(puntos[i][2] for i in HOMBROS + CADERAS) < CONF_PUNTO:
        return None
    hx, hy = (puntos[5][0] + puntos[6][0]) / 2, (puntos[5][1] + puntos[6][1]) / 2
    cx, cy = (puntos[11][0] + puntos[12][0]) / 2, (puntos[11][1] + puntos[12][1]) / 2
    # Ángulo entre el torso y la vertical: 0° de pie, 90° acostado, más de 90° cabeza abajo.
    # shortcut: alguien tirado a lo largo del eje de la cámara se ve "vertical" en la imagen;
    # para eso hace falta saber la altura de la cámara o una segunda cámara.
    return math.degrees(math.atan2(abs(cx - hx), cy - hy)) > ANGULO_CAIDO
