"""Detector de cabezas con y sin casco (helmet_detector.py) y su uso en /predict."""
import base64
import os
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
    assert r.get_json() == {"has_face": True, "modos": {"casco": True, "chaleco": True},
                            "faces": [{"con_casco": 25.0, "sin_casco": 75.0, "bbox": [10, 20, 30, 40],
                                       "nombre": None, "chaleco": False}]}
    crop_head.assert_not_called()


CABEZA = {"bbox": [10, 20, 30, 40], "clase": "casco", "conf": 0.9, "p_casco": 0.9, "p_sin": 0.1}


def _predict_con_config(flask_app, operator_client, **cfg):
    import data_store
    data_store.save_config({**data_store.DEFAULT_CONFIG, **cfg})
    with mock.patch.object(flask_app.helmet_detector, "disponible", return_value=True),          mock.patch.object(flask_app.helmet_detector, "detectar", return_value=[CABEZA]):
        return operator_client.post("/predict", json={"image": _frame_b64()}).get_json()


def test_solo_casco_no_evalua_chaleco(flask_app, operator_client):
    with mock.patch.object(flask_app.epp, "chalecos_por_cabeza") as chalecos:
        data = _predict_con_config(flask_app, operator_client, detectar_chaleco=False)
    chalecos.assert_not_called()
    assert data["modos"] == {"casco": True, "chaleco": False}
    assert "chaleco" not in data["faces"][0] and data["faces"][0]["con_casco"] == 90.0


def test_solo_chaleco_usa_el_modelo_de_chalecos_sin_cascos(flask_app, operator_client):
    persona = {"bbox": [5, 5, 100, 300], "chaleco": False}
    with mock.patch.object(flask_app.epp, "disponible", return_value=True),          mock.patch.object(flask_app.epp, "detectar_personas", return_value=[persona]):
        data = _predict_con_config(flask_app, operator_client, detectar_casco=False)
    assert data["modos"] == {"casco": False, "chaleco": True}
    assert data["faces"] == [{"bbox": [5, 5, 100, 300], "chaleco": False, "nombre": None}]


def test_solo_chaleco_sin_modelo_usa_las_cabezas_sin_informar_casco(flask_app, operator_client):
    with mock.patch.object(flask_app.epp, "disponible", return_value=False),          mock.patch.object(flask_app.epp, "chalecos_por_cabeza", return_value=[True]):
        data = _predict_con_config(flask_app, operator_client, detectar_casco=False)
    assert data["faces"] == [{"bbox": [10, 20, 30, 40], "nombre": None, "chaleco": True}]


def test_config_vieja_sin_claves_nuevas_activa_las_dos(tmp_data):
    import data_store
    data_store.save_config({"min_confidence": 60})
    cfg = data_store.load_config()
    assert (cfg["detectar_casco"], cfg["detectar_chaleco"], cfg["min_confidence"]) == (True, True, 60)


def test_openvino_y_cv2_dan_las_mismas_detecciones():
    if not helmet_detector.disponible():
        pytest.skip("falta detector_casco.onnx")
    img = cv2.imread(os.path.join(os.path.dirname(helmet_detector.MODEL_PATH), "con_casco", "1.jpg"))
    img = img if img is not None else np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    cv_net = cv2.dnn.readNetFromONNX(helmet_detector.MODEL_PATH)
    a = helmet_detector.correr_yolo(cv_net, helmet_detector._lock, img, 2)
    fp32, _ = helmet_detector.cargar_modelo(helmet_detector.MODEL_PATH, int8=False)
    b = helmet_detector.correr_yolo(fp32, helmet_detector._lock, img, 2)
    assert len(a) == len(b)
    assert np.allclose([bb for bb, _ in a], [bb for bb, _ in b], atol=3)
    # INT8 (cuantizar.py): mismas cabezas y clases, cajas a menos del 1 % del tamaño de la foto
    c = helmet_detector.correr_yolo(helmet_detector._net, helmet_detector._lock, img, 2)
    assert [s.argmax() for _, s in a] == [s.argmax() for _, s in c]
    assert np.allclose([bb for bb, _ in a], [bb for bb, _ in c], atol=0.01 * max(img.shape[:2]))


def test_cargar_modelo_vuelve_a_cv2_si_openvino_falla(monkeypatch):
    core = mock.MagicMock()
    core.compile_model.side_effect = RuntimeError("sin plugin CPU")
    monkeypatch.setattr(helmet_detector, "_core", core)
    monkeypatch.setattr(helmet_detector.cv2.dnn, "readNetFromONNX", lambda p: "red-cv2")
    assert helmet_detector.cargar_modelo("x.onnx") == ("red-cv2", "cv2.dnn")
