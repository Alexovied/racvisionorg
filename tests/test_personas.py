"""Registro de personas para reconocimiento facial desde el panel (admin.personas)."""
import io
import os

import cv2
import numpy as np
import pytest

import data_store
import face_id


@pytest.fixture
def admin_client(flask_app):
    data_store.create_user("admin_test", "clave-admin", role="admin")
    c = flask_app.app.test_client()
    assert c.post("/admin/login", data={"username": "admin_test", "password": "clave-admin"}).status_code == 302
    return c


@pytest.fixture
def sface_falso(monkeypatch):
    # Una foto con algo claro "tiene cara"; una foto negra no
    monkeypatch.setattr(face_id, "disponible", lambda: True)
    monkeypatch.setattr(face_id, "embedding", lambda img: np.ones(128) / np.sqrt(128) if img.mean() > 10 else None)


def _jpg(claro=True):
    img = np.full((120, 120, 3), 200 if claro else 0, dtype=np.uint8)
    return io.BytesIO(cv2.imencode(".jpg", img)[1].tobytes())


def _alta(c, nombre="María Gómez", fotos=None, consentimiento="si"):
    fotos = fotos if fotos is not None else [_jpg() for _ in range(3)]
    data = {"nombre": nombre, "consentimiento": consentimiento, "fotos": [(f, f"f{i}.jpg") for i, f in enumerate(fotos)]}
    return c.post("/admin/personas", data=data, content_type="multipart/form-data", follow_redirects=True)


def test_alta_guarda_embeddings_y_no_las_fotos(admin_client, sface_falso):
    r = _alta(admin_client)
    assert "quedó registrada con 3 fotos" in r.get_data(as_text=True)
    [p] = face_id.listar()
    assert (p["nombre"], p["fotos"], p["por"], p["consentimiento"]) == ("María Gómez", 3, "admin_test", True)
    archivos = [f for _, _, fs in os.walk(data_store.DATA_DIR) for f in fs]
    assert not [f for f in archivos if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]


@pytest.mark.parametrize("cambio, mensaje", [
    ({"consentimiento": ""}, "Falta confirmar el consentimiento"),
    ({"nombre": "R2D2"}, "nombre válido"),
    ({"fotos": [_jpg(), _jpg()]}, "entre 3 y 10 fotos"),
    ({"fotos": [_jpg(), _jpg(), io.BytesIO(b"no soy una imagen")]}, "no es una imagen"),
    ({"fotos": [_jpg(), _jpg(False), _jpg(False)]}, "Solo se encontró la cara en 1 de 3"),
])
def test_alta_rechazada_no_guarda_nada(admin_client, sface_falso, cambio, mensaje):
    assert mensaje in _alta(admin_client, **cambio).get_data(as_text=True)
    assert face_id.listar() == []


def test_no_duplica_y_se_puede_borrar(admin_client, sface_falso):
    _alta(admin_client)
    assert "ya está registrada" in _alta(admin_client, nombre="maría gómez").get_data(as_text=True)
    r = admin_client.post("/admin/personas/borrar", data={"nombre": "María Gómez"}, follow_redirects=True)
    assert "Se borraron los datos" in r.get_data(as_text=True)
    assert face_id.listar() == [] and data_store._load_json("personas_registro.json", {}) == {}


def test_solo_admin(operator_client, sface_falso):
    for r in (operator_client.get("/admin/personas"), _alta(operator_client), operator_client.get("/admin/personas/consentimiento")):
        assert "No tienes permisos" in r.get_data(as_text=True) or r.status_code in (302, 403)
    assert face_id.listar() == []


def test_formulario_de_consentimiento(admin_client):
    html = admin_client.get("/admin/personas/consentimiento").get_data(as_text=True)
    assert "Consentimiento para el uso de reconocimiento facial" in html and "Ley 25.326" in html
