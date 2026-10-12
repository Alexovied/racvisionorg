"""
Fixtures de pytest para RACVision.

- TensorFlow/Keras se reemplaza por mocks ANTES de importar app.py: los tests
  no cargan modelo.keras ni necesitan TensorFlow instalado.
- El puerto serie (pyserial) también es un mock que siempre falla al abrir:
  se prueba el modo "sin ESP32 conectado" y nunca se toca un COM real.
- data_store usa una carpeta temporal: nunca se leen ni escriben los
  archivos reales de data/ (users.json, .secret_key, events.json, ...).
"""
import os
import sys
from unittest import mock

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Variables de entorno para que app.py no lea data/.secret_key ni genere un admin.
os.environ["SECRET_KEY"] = "clave-solo-para-tests"
os.environ["ADMIN_PASSWORD"] = "admin-solo-para-tests"
os.environ["N8N_WEBHOOK_URL"] = ""

# ── Mock de Keras / TensorFlow ───────────────────────────────────────────────
_keras = mock.MagicMock(name="keras")
_mobilenet = mock.MagicMock(name="keras.applications.mobilenet_v2")
# preprocess_input de MobileNetV2 (x/127.5 - 1); el mock lo replica para poder
# verificar que app.py sigue llamándolo con la entrada correcta.
_mobilenet.preprocess_input = mock.MagicMock(side_effect=lambda x: x / 127.5 - 1.0)
_keras.applications.mobilenet_v2 = _mobilenet


class _FakeGlorotUniform:  # app.py le aplica un patch a __init__
    def __init__(self, seed=None, **kwargs):
        self.seed = seed


_keras.initializers.GlorotUniform = _FakeGlorotUniform


class _FakeLayer:  # app.py también parchea BatchNormalization y Dense
    def __init__(self, *args, **kwargs):
        pass


_keras.layers.BatchNormalization = type("BatchNormalization", (_FakeLayer,), {})
_keras.layers.Dense = type("Dense", (_FakeLayer,), {})
sys.modules["keras"] = _keras
sys.modules["keras.applications"] = _keras.applications
sys.modules["keras.applications.mobilenet_v2"] = _mobilenet

# ── Mock del puerto serie: abrir siempre falla (ESP32 desconectado) ──────────
_serial = mock.MagicMock(name="serial")
_serial.Serial.side_effect = OSError("puerto serie simulado: no conectado")
sys.modules["serial"] = _serial

import data_store  # noqa: E402

_REAL_DATA_DIR = data_store.DATA_DIR


@pytest.fixture(scope="session")
def flask_app():
    import app as app_module
    app_module.app.config.update(TESTING=True)
    # Si en el futuro se agrega CSRF, los tests de API lo desactivan acá.
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    return app_module


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(data_store, "DATA_DIR", str(data_dir))
    assert data_store.DATA_DIR != _REAL_DATA_DIR
    return data_dir


@pytest.fixture
def client(flask_app):
    return flask_app.app.test_client()


@pytest.fixture
def operator_client(flask_app):
    data_store.create_user("operador_test", "clave-operador", role="operator")
    c = flask_app.app.test_client()
    r = c.post("/admin/login", data={"username": "operador_test", "password": "clave-operador"})
    assert r.status_code == 302
    return c


@pytest.fixture
def fake_model(flask_app, monkeypatch):
    """Modelo falso que devuelve [con_casco, sin_casco] = [0.9, 0.1] por cara.
    Desactiva el detector YOLO para probar la ruta de respaldo (cara + Keras)
    aunque detector_casco.onnx esté en la carpeta."""
    monkeypatch.setattr(flask_app.helmet_detector, "disponible", lambda: False)
    model = mock.MagicMock(name="model")

    def _predict(batch, training=False):
        out = mock.MagicMock()
        out.numpy.return_value = np.tile(np.array([[0.9, 0.1]], dtype="float32"), (len(batch), 1))
        return out

    model.side_effect = _predict
    monkeypatch.setattr(flask_app, "model", model)
    return model
