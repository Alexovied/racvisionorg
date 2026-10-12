"""Detector de cabezas con y sin casco (helmet_detector.py) y su uso en /predict."""
import base64
from unittest import mock

import cv2
import numpy as np
import pytest

import helmet_detector


def _frame_b64():
    ok, buf = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class _RedFalsa:
    """Imita cv2.dnn.Net devolviendo una salida YOLOv8 (1, 6, N) armada a mano."""

    def __init__(self, filas):
        out = np.zeros((1, 6, 8400), dtype=np.float32)
        for i, fila in enumerate(filas):
            out[0, :, i] = fila
        self.out = out

    def setInput(self, blob):
        assert blob.shape == (1, 3, 640, 640)

    def forward(self):
        return self.out


def test_detectar_convierte_cajas_a_la_imagen_original(monkeypatch):
    # Imagen 640x480: escala 1, relleno de 80 px arriba y abajo en el canvas de 640x640.
    # Fila YOLO = cx, cy, w, h (en el canvas), puntaje casco, puntaje sin casco
    filas = [
        [320, 180, 50, 60, 0.90, 0.05],   # cabeza con casco
        [322, 181, 50, 60, 0.80, 0.10],   # misma cabeza, la elimina el NMS
        [100, 300, 40, 40, 0.02, 0.70],   # otra cabeza, sin casco
        [500, 400, 40, 40, 0.10, 0.20],   # debajo del umbral, se descarta
    ]
    monkeypatch.setattr(helmet_detector, "_net", _RedFalsa(filas))
    res = helmet_detector.detectar(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(res) == 2
    casco = next(r for r in res if r["clase"] == "casco")
    assert casco["bbox"] == [295, 70, 50, 60]
    assert casco["p_casco"] == pytest.approx(0.90 / 0.95)
    assert casco["p_casco"] + casco["p_sin"] == pytest.approx(1)
    sin = next(r for r in res if r["clase"] == "sin_casco")
    assert sin["bbox"] == [80, 200, 40, 40]


def test_detectar_sin_modelo_devuelve_vacio(monkeypatch):
    monkeypatch.setattr(helmet_detector, "_net", None)
    assert not helmet_detector.disponible()
    assert helmet_detector.detectar(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_predict_usa_detector_de_cabezas_si_esta_disponible(flask_app, operator_client):
    cabeza = {"bbox": [10, 20, 30, 40], "clase": "sin_casco", "conf": 0.8, "p_casco": 0.25, "p_sin": 0.75}
    with mock.patch.object(flask_app.helmet_detector, "disponible", return_value=True), \
         mock.patch.object(flask_app.helmet_detector, "detectar", return_value=[cabeza]), \
         mock.patch.object(flask_app, "crop_head") as crop_head:
        r = operator_client.post("/predict", json={"image": _frame_b64()})
    assert r.status_code == 200
    assert r.get_json() == {"has_face": True,
                            "faces": [{"con_casco": 25.0, "sin_casco": 75.0, "bbox": [10, 20, 30, 40]}]}
    crop_head.assert_not_called()
