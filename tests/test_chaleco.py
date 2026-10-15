"""Chaleco reflectante: color de alta visibilidad en el torso y alertas sin_chaleco."""
import json

import numpy as np

import epp

CABEZA = [100, 40, 60, 60]          # x, y, w, h en una imagen de 480x640
AMARILLO_FLUOR = (40, 255, 200)     # BGR del chaleco de las fotos de prueba


def _imagen(color_torso=None):
    img = np.full((480, 640, 3), (180, 180, 180), np.uint8)
    if color_torso:
        img[110:250, 70:190] = color_torso
    return img


def test_torso_amarillo_fluor_es_chaleco():
    assert epp.tiene_chaleco(_imagen(AMARILLO_FLUOR), CABEZA) is True


def test_torso_gris_no_es_chaleco():
    assert epp.tiene_chaleco(_imagen(), CABEZA) is False


def test_sin_torso_a_la_vista_no_se_juzga():
    # cabeza pegada al borde inferior: no entra el torso
    assert epp.tiene_chaleco(_imagen(AMARILLO_FLUOR), [100, 400, 60, 60]) is None


def test_con_modelo_la_persona_de_cada_cabeza_lleva_o_no_chaleco(monkeypatch):
    persona_a, chaleco_a = [80, 30, 120, 300], [95, 120, 90, 100]
    persona_b = [300, 30, 120, 300]
    dets = [(persona_a, np.array([0.9, 0.1])), (chaleco_a, np.array([0.1, 0.8])), (persona_b, np.array([0.9, 0.1]))]
    monkeypatch.setattr(epp, "_net", object())
    monkeypatch.setattr(epp.helmet_detector, "correr_yolo", lambda *a, **k: dets)
    cabezas = [[110, 40, 60, 60], [330, 40, 60, 60], [520, 40, 60, 60]]
    # A con chaleco, B sin chaleco, la tercera sin persona detectada → regla de color (torso gris)
    assert epp.chalecos_por_cabeza(_imagen(), cabezas) == [True, False, False]


def test_evento_sin_chaleco_es_alerta_con_persona(operator_client, tmp_data):
    r = operator_client.post("/admin/api/events", json={"type": "sin_chaleco", "confidence": 88, "persona": "luciano"})
    assert r.status_code == 200
    ev = json.loads((tmp_data / "events.json").read_text())[0]
    assert (ev["type"], ev["persona"]) == ("sin_chaleco", "luciano")
    assert operator_client.post("/api/alarm", json={"state": "sin_chaleco"}).status_code == 200


def test_detectar_personas_con_modelo_o_color(monkeypatch):
    persona_a, chaleco_a = [80, 30, 120, 300], [95, 120, 90, 100]
    persona_b = [300, 30, 120, 300]
    dets = [(persona_a, np.array([0.9, 0.1])), (chaleco_a, np.array([0.1, 0.8])), (persona_b, np.array([0.9, 0.1]))]
    monkeypatch.setattr(epp, "_net", object())
    monkeypatch.setattr(epp.helmet_detector, "correr_yolo", lambda *a, **k: dets)
    img = _imagen()
    img[100:200, 320:400] = AMARILLO_FLUOR    # B: el modelo no ve chaleco pero la regla de color sí
    assert [p["chaleco"] for p in epp.detectar_personas(img)] == [True, True]
    assert [p["chaleco"] for p in epp.detectar_personas(_imagen())] == [True, False]


def test_detectar_personas_sin_modelo_devuelve_vacio(monkeypatch):
    monkeypatch.setattr(epp, "_net", None)
    assert epp.detectar_personas(_imagen()) == []


def test_config_no_deja_apagar_las_dos_detecciones(flask_app):
    import data_store
    data_store.create_user("admin_test", "clave-admin", role="admin")
    c = flask_app.app.test_client()
    c.post("/admin/login", data={"username": "admin_test", "password": "clave-admin"})
    form = {"action": "save", "min_confidence": 51, "confirm_threshold": 3, "scan_rate_ms": 100}
    c.post("/admin/config", data={**form, "detectar_chaleco": "on"})
    cfg = data_store.load_config()
    assert (cfg["detectar_casco"], cfg["detectar_chaleco"]) == (False, True)
    c.post("/admin/config", data=form)     # las dos apagadas: se rechaza y queda lo anterior
    cfg = data_store.load_config()
    assert (cfg["detectar_casco"], cfg["detectar_chaleco"]) == (False, True)
