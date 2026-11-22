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
    assert (tmp_data / "alertas" / f"{ev['id']}.webp").exists()
    img = operator_client.get(f"/admin/alertas/{ev['id']}.webp")
    assert img.status_code == 200 and img.mimetype == "image/webp"
    assert img.data[:4] == b"RIFF" and img.data[8:12] == b"WEBP"
    assert cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR) is not None


def test_alerta_vieja_en_jpg_se_sigue_viendo(operator_client, tmp_data):
    """Las fotos guardadas antes del cambio a WebP siguen saliendo por el mismo enlace."""
    ev = data_store.append_event("sin_casco", 80, "webcam")
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), 90, np.uint8))
    (tmp_data / "alertas").mkdir(exist_ok=True)
    (tmp_data / "alertas" / f"{ev['id']}.jpg").write_bytes(buf.tobytes())
    for ext in ("jpg", "webp"):
        img = operator_client.get(f"/admin/alertas/{ev['id']}.{ext}")
        assert img.status_code == 200 and img.mimetype == "image/jpeg"


def test_motor_guarda_la_foto_jpeg_como_webp(tmp_data):
    import admin
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), 60, np.uint8))
    ev = admin.registrar_evento("sin_chaleco", 90, "webcam", None, buf.tobytes())
    data = (tmp_data / "alertas" / f"{ev['id']}.webp").read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    assert not (tmp_data / "alertas" / f"{ev['id']}.jpg").exists()


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
    assert client.get(f"/admin/alertas/{ev['id']}.webp").status_code in (302, 401)
    assert operator_client.get("/admin/alertas/..%2F..%2Fusers.jpg").status_code == 404
