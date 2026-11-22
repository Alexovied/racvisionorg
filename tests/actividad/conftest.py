"""Configuración de los tests de la Actividad Grupal (Evaluación de Proyectos).

Reutiliza las fixtures de tests/conftest.py (Flask con Keras y puerto serie
simulados, datos en una carpeta temporal) y suma:
- el marcador `defecto`: tests que documentan un defecto encontrado (fallan a
  propósito hasta que se corrija). Para ver solo la suite "verde":
      pytest tests/actividad -m "not defecto"
- fixtures de login por rol.
"""
import pytest

import data_store


def pytest_configure(config):
    config.addinivalue_line("markers", "defecto(id): test que documenta un defecto encontrado (DEF-xx)")
    config.addinivalue_line("markers", "caja_blanca: técnica de caja blanca")
    config.addinivalue_line("markers", "caja_negra: técnica de caja negra")
    config.addinivalue_line("markers", "caja_gris: técnica híbrida (caja gris)")


def login(flask_app, usuario, rol):
    """Crea un usuario con ese rol y devuelve un cliente HTTP con la sesión abierta."""
    data_store.create_user(usuario, "clave-" + usuario, role=rol)
    c = flask_app.app.test_client()
    r = c.post("/admin/login", data={"username": usuario, "password": "clave-" + usuario})
    assert r.status_code == 302
    return c


@pytest.fixture
def admin_c(flask_app):
    return login(flask_app, "admin_qa", "admin")


@pytest.fixture
def viewer_c(flask_app):
    return login(flask_app, "viewer_qa", "viewer")


@pytest.fixture
def sin_propagar(flask_app, monkeypatch):
    """Como en producción: una excepción no atrapada se convierte en HTTP 500
    (con TESTING=True Flask la relanza y no se ve el código real)."""
    monkeypatch.setitem(flask_app.app.config, "TESTING", False)
    monkeypatch.setitem(flask_app.app.config, "PROPAGATE_EXCEPTIONS", False)
