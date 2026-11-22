"""Detección de rostros robusta al fondo.

Detector principal: YuNet (cv2.FaceDetectorYN), una red neuronal pequeña que
viene con OpenCV >= 4.5.4. A diferencia de las cascadas Haar, no se confunde
con fondos con rayas, orejas o muebles, y cada detección trae un puntaje de
confianza que permite descartar las dudosas.

Si el archivo .onnx no está o OpenCV no trae FaceDetectorYN, se usan las
cascadas Haar de antes como respaldo.
"""
import os
import threading
import cv2
import numpy as np

import helmet_detector  # comparte el Core y el dispositivo de OpenVINO

YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_detection_yunet_2023mar.onnx")

# Puntaje mínimo para aceptar una cara. En el dataset las caras reales dan
# 0.8-0.9 incluso lejos de la cámara; nucas y objetos del fondo dieron 0.5-0.68.
SCORE_THRESHOLD = 0.7
# El reintento con CLAHE realza también texturas del fondo, así que exige más
CLAHE_SCORE_THRESHOLD = 0.8
NMS_THRESHOLD   = 0.3
MAX_DIM         = 800
# Lado mínimo de una cara (px de la imagen reducida). YuNet detecta bien caras
# lejanas de ~12 px, así que el filtro solo descarta cajas degeneradas.
MIN_FACE        = 8

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

_yunet = None
_yunet_lock = threading.Lock()  # setInputSize/detect modifican estado interno
if hasattr(cv2, "FaceDetectorYN") and os.path.exists(YUNET_PATH):
    try:
        _yunet = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320),
                                           SCORE_THRESHOLD, NMS_THRESHOLD, 50)
        print("[+] Detector de rostros: YuNet")
    except cv2.error as e:
        print(f"[!] No se pudo cargar YuNet ({e}). Se usan cascadas Haar.")
if _yunet is None:
    print("[!] Detector de rostros: cascadas Haar (YuNet no disponible)")


class _YuNetOpenVINO:
    """YuNet con OpenVINO, con la misma interfaz que cv2.FaceDetectorYN (setInputSize,
    setScoreThreshold, detect) y su mismo pre y posprocesamiento (opencv face_detect.cpp)."""

    def __init__(self, path, score, nms, top_k):
        self.modelo, self.score, self.nms, self.top_k = helmet_detector._core.read_model(path), score, nms, top_k
        self.redes, self.tam = {}, (320, 320)

    def setInputSize(self, tam):
        self.tam = tuple(tam)

    def setScoreThreshold(self, score):
        self.score = score

    def _red(self, pw, ph):
        # Una red compilada por tamaño (la cámara casi siempre manda el mismo)
        if (pw, ph) not in self.redes:
            m = self.modelo.clone()
            m.reshape([1, 3, ph, pw])
            self.redes[(pw, ph)] = helmet_detector._core.compile_model(m, helmet_detector.DEVICE)
        return self.redes[(pw, ph)]

    def detect(self, img):
        w, h = self.tam
        pw, ph = ((w - 1) // 32 + 1) * 32, ((h - 1) // 32 + 1) * 32
        blob = cv2.dnn.blobFromImage(cv2.copyMakeBorder(img, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0))
        red = self._red(pw, ph)
        res = red(blob)
        out = {n: res[red.output(n)][0] for n in (f"{k}_{s}" for k in ("cls", "obj", "bbox", "kps") for s in (8, 16, 32))}
        filas = []
        for s in (8, 16, 32):
            cols = pw // s
            idx = np.arange(out[f"cls_{s}"].shape[0])
            c, r = (idx % cols)[:, None], (idx // cols)[:, None]
            score = np.sqrt(np.clip(out[f"cls_{s}"], 0, 1) * np.clip(out[f"obj_{s}"], 0, 1))
            b = out[f"bbox_{s}"]
            wh = np.exp(b[:, 2:4]) * s
            xy = (np.hstack([c, r]) + b[:, 0:2]) * s - wh / 2
            kps = (out[f"kps_{s}"].reshape(-1, 5, 2) + np.hstack([c, r])[:, None, :]) * s
            filas.append(np.hstack([xy, wh, kps.reshape(-1, 10), score]))
        filas = np.vstack(filas).astype(np.float32)
        keep = cv2.dnn.NMSBoxes(filas[:, :4].tolist(), filas[:, 14].tolist(), self.score, self.nms, top_k=self.top_k)
        return 1, (filas[np.array(keep).flatten()] if len(keep) else None)


if _yunet is not None and helmet_detector._core is not None:
    try:
        _yunet = _YuNetOpenVINO(YUNET_PATH, SCORE_THRESHOLD, NMS_THRESHOLD, 50)
        print("[+] YuNet corre con OpenVINO")
    except Exception as e:
        print(f"[!] YuNet sigue con cv2.dnn ({e})")

_face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


def _equalize(img_bgr):
    """CLAHE sobre el canal L para mejorar contraste con poca luz."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([_clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def _yunet_detect(img):
    with _yunet_lock:
        _yunet.setInputSize((img.shape[1], img.shape[0]))
        _, faces = _yunet.detect(img)
        if faces is None:
            # Segundo intento con contraste realzado (baja luz / minería)
            _yunet.setScoreThreshold(CLAHE_SCORE_THRESHOLD)
            try:
                _, faces = _yunet.detect(_equalize(img))
            finally:
                _yunet.setScoreThreshold(SCORE_THRESHOLD)
    if faces is None:
        return []
    ih, iw = img.shape[:2]
    boxes = []
    for f in faces:
        x, y, w, h = (int(round(v)) for v in f[:4])
        x, y = max(0, x), max(0, y)
        w, h = min(w, iw - x), min(h, ih - y)
        if w >= MIN_FACE and h >= MIN_FACE:
            boxes.append([x, y, w, h])
    return boxes


def yunet_rows(img):
    """Filas crudas de YuNet (caja + 5 puntos de la cara + puntaje), las que
    necesita FaceRecognizerSF.alignCrop. None si no hay YuNet o no hay caras."""
    if _yunet is None or img is None or min(img.shape[:2]) < 20:
        return None
    with _yunet_lock:
        _yunet.setInputSize((img.shape[1], img.shape[0]))
        _, faces = _yunet.detect(img)
    return faces


def _safe_haar(cascade, gray, min_neighbors=3, min_size=(40, 40)):
    h, w = gray.shape[:2]
    if h < min_size[0] * 2 or w < min_size[0] * 2:
        return []
    try:
        return [list(f) for f in cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=min_size)]
    except cv2.error:
        return []


def _strict_profile(img):
    """Perfil puro: YuNet casi no lo ve. Haar de perfil con minNeighbors=5
    no dio ningún falso positivo sobre 31 fotos de nucas y fondos cargados."""
    iw = img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _safe_haar(_profile_cascade, gray, min_neighbors=5)
    if not faces:
        faces = [[iw - (x + w), y, w, h]
                 for x, y, w, h in _safe_haar(_profile_cascade, cv2.flip(gray, 1), min_neighbors=5)]
    return faces


def _haar_detect(img):
    """Lógica anterior basada en cascadas Haar (solo como respaldo)."""
    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _safe_haar(_face_cascade, gray)
    if not faces:
        faces = _safe_haar(_profile_cascade, gray)
    if not faces:
        faces = [[iw - (x + w), y, w, h]
                 for x, y, w, h in _safe_haar(_profile_cascade, cv2.flip(gray, 1))]
    if not faces:
        gray_eq = _clahe.apply(gray)
        faces = (_safe_haar(_face_cascade, gray_eq, 2, (30, 30))
                 or _safe_haar(_profile_cascade, gray_eq, 2, (30, 30)))
    # Haar da muchos falsos positivos en la parte baja de la imagen
    return [f for f in faces if f[1] + f[3] / 2 < ih * 0.75]


def detect_faces(img):
    """Devuelve las caras como [x, y, w, h] en coordenadas de `img`,
    ordenadas de mayor a menor tamaño."""
    if img is None:
        return []
    h_orig, w_orig = img.shape[:2]
    if h_orig < 60 or w_orig < 60:
        return []

    scale = min(1.0, MAX_DIM / max(h_orig, w_orig))
    small = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale))) if scale < 1.0 else img

    if _yunet is not None:
        boxes = _yunet_detect(small) or _strict_profile(small)
    else:
        boxes = _haar_detect(small)
    boxes = [[int(v / scale) for v in b] for b in boxes]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
