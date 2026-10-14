"""Alertas sin casco: se guardan con foto, persona y ubicación."""
import base64
import json

import cv2
import numpy as np

import data_store


def _foto():
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), 120, np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _eventos(tmp_data):
    return json.loads((tmp_data / "events.json").read_text())


def test_alerta_guarda_foto_persona_y_ubicacion(operator_client, tmp_data):
    cam = data_store.create_camera("Taller", "Sector B", "webcam")
    r = operator_client.post("/admin/api/events", json={
        "type": "sin_casco", "confidence": 91, "camera": cam["id"], "persona": "owen", "foto": _foto()})
    assert r.status_code == 200

    ev = _eventos(tmp_data)[0]
    assert (ev["persona"], ev["camera"], ev["ubicacion"], ev["foto"]) == ("owen", "Taller", "Sector B", True)
    img = operator_client.get(f"/admin/alertas/{ev['id']}.jpg")
    assert img.status_code == 200 and img.mimetype == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR) is not None


def test_alerta_sin_nombre_ni_foto_valida(operator_client, tmp_data):
    r = operator_client.post("/admin/api/events", json={
        "type": "sin_casco", "confidence": 80, "camera": "webcam", "foto": "data:image/jpeg;base64,bm8gZXMgdW5hIGltYWdlbg=="})
    assert r.status_code == 200
    ev = _eventos(tmp_data)[0]
    assert ev["persona"] == "Sin identificar" and ev["camera"] == "Detector web"
    assert "foto" not in ev


def test_evento_con_casco_no_guarda_foto_ni_persona(operator_client, tmp_data):
    operator_client.post("/admin/api/events", json={
        "type": "con_casco", "confidence": 90, "camera": "webcam", "persona": "owen", "foto": _foto()})
    ev = _eventos(tmp_data)[0]
    assert "foto" not in ev and "persona" not in ev


def test_foto_de_alerta_requiere_sesion_y_id_valido(client, operator_client, tmp_data):
    operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80, "foto": _foto()})
    ev = _eventos(tmp_data)[0]
    assert client.get(f"/admin/alertas/{ev['id']}.jpg").status_code in (302, 401)
    assert operator_client.get("/admin/alertas/..%2F..%2Fusers.jpg").status_code == 404
