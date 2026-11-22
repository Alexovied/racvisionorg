"""Modo anónimo y cifrado de los datos biométricos (Ley 25.326)."""
import json
from unittest import mock

import cv2
import numpy as np
import pytest

import admin
import data_store
import face_id


def _foto(color=120):
    ok, jpg = cv2.imencode(".jpg", np.full((200, 200, 3), color, np.uint8))
    return jpg.tobytes()


def test_anonimizar_pixela_las_caras(monkeypatch):
    img = np.zeros((200, 200, 3), np.uint8)
    img[60:120, 60:120] = np.random.default_rng(1).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    ok, jpg = cv2.imencode(".png", img)
    monkeypatch.setattr(admin.face_detector, "detect_faces", lambda im: [[70, 70, 40, 40]])
    salida = cv2.imdecode(np.frombuffer(admin.anonimizar(jpg.tobytes()), np.uint8), cv2.IMREAD_COLOR)
    cara = salida[80:100, 80:100].astype(int)
    # pixelado: bloques grandes de un solo color en lugar del ruido original
    assert np.abs(np.diff(cara, axis=1)).mean() < np.abs(np.diff(img[80:100, 80:100].astype(int), axis=1)).mean() / 3
    assert admin.anonimizar(b"no-es-imagen") is None


def test_modo_anonimo_no_guarda_nombres_y_pixela(flask_app, monkeypatch):
    cfg = data_store.load_config()
    cfg["modo_anonimo"] = True
    data_store.save_config(cfg)
    pixeladas = []
    monkeypatch.setattr(admin, "anonimizar", lambda f: pixeladas.append(f) or _foto(50))
    with mock.patch("admin.enviar_comando_esp32"):
        ev = admin.registrar_evento("sin_casco", 90, "webcam", "José", _foto())
    assert ev["persona"] == "Sin identificar" and ev["foto"] is True and len(pixeladas) == 1


@pytest.fixture
def admin_client(flask_app):
    data_store.create_user("admin_priv", "clave-admin", role="admin")
    c = flask_app.app.test_client()
    assert c.post("/admin/login", data={"username": "admin_priv", "password": "clave-admin"}).status_code == 302
    return c


def test_config_guarda_el_modo_anonimo(admin_client):
    form = {"action": "save", "min_confidence": 60, "confirm_threshold": 3, "scan_rate_ms": 100,
            "detectar_casco": "on", "modo_anonimo": "on"}
    admin_client.post("/admin/config", data=form)
    assert data_store.load_config()["modo_anonimo"] is True


def test_conocidos_vacios_en_modo_anonimo(flask_app, operator_client, monkeypatch):
    monkeypatch.setattr(face_id, "_personas", lambda: [("José", np.ones((1, 128), np.float32))])
    cfg = data_store.load_config()
    cfg["modo_anonimo"] = True
    data_store.save_config(cfg)
    with flask_app.app.test_request_context():
        from flask_login import login_user
        login_user(admin.User(data_store.get_user_by_username("operador_test")))
        assert flask_app._conocidos() == []


# ── Cifrado de embeddings ────────────────────────────────────────────────────

@pytest.fixture
def clave(monkeypatch):
    from cryptography.fernet import Fernet
    k = Fernet.generate_key().decode()
    monkeypatch.setenv(face_id.CLAVE_ENV, k)
    return k


def test_embeddings_quedan_cifrados_en_disco(clave):
    face_id._escribir_db({"José": [[0.1, 0.2]]})
    crudo = json.loads(open(face_id._ruta_db(), encoding="utf-8").read())
    assert set(crudo) == {"cifrado"} and "José" not in json.dumps(crudo)
    assert face_id._leer_db() == {"José": [[0.1, 0.2]]}


def test_archivo_viejo_sin_cifrar_se_lee_y_se_cifra_al_guardar(clave):
    data_store._save_json("personas.json", {"Ana": [[0.3]]})
    datos = face_id._leer_db()
    assert datos == {"Ana": [[0.3]]}
    face_id._escribir_db(datos)
    assert set(data_store._load_json("personas.json", {})) == {"cifrado"}


def test_sin_la_clave_nadie_se_reconoce_y_no_se_borra_nada(clave, monkeypatch):
    face_id._escribir_db({"José": [[0.1]]})
    monkeypatch.delenv(face_id.CLAVE_ENV)
    assert face_id._leer_db() == {}
    assert set(data_store._load_json("personas.json", {})) == {"cifrado"}     # sigue ahí


def test_sin_clave_no_se_puede_registrar_ni_pisar_el_archivo(clave, monkeypatch):
    face_id._escribir_db({"José": [[0.1]]})
    monkeypatch.delenv(face_id.CLAVE_ENV)
    monkeypatch.setattr(face_id, "_embeddings", lambda imgs: ([[0.2]] * 3, 0))
    with pytest.raises(RuntimeError):
        face_id.registrar("Ana", [None] * 3, "admin")
    with pytest.raises(RuntimeError):
        face_id.borrar("José")
    assert set(data_store._load_json("personas.json", {})) == {"cifrado"}


def test_modo_anonimo_sin_caras_detectadas_no_guarda_foto(monkeypatch):
    monkeypatch.setattr(admin.face_detector, "detect_faces", lambda im: [])
    assert admin.anonimizar(_foto()) is None
