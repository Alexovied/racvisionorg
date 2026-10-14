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


def test_evento_sin_chaleco_es_alerta_con_persona(operator_client, tmp_data):
    r = operator_client.post("/admin/api/events", json={"type": "sin_chaleco", "confidence": 88, "persona": "luciano"})
    assert r.status_code == 200
    ev = json.loads((tmp_data / "events.json").read_text())[0]
    assert (ev["type"], ev["persona"]) == ("sin_chaleco", "luciano")
    assert operator_client.post("/api/alarm", json={"state": "sin_chaleco"}).status_code == 200
