"""CAJA GRIS (híbridas) — casos funcionales diseñados con conocimiento parcial
de la arquitectura de RAC Vision:

    navegador ──HTTP──> app.py / admin.py (Flask) ──> data_store.py ──> data/*.json
                                    │                        └──> data/alertas/<id>.jpg
                                    ├──> hardware.py ──serial──> ESP32 (luz + buzzer)
                                    └──> webhook n8n (HTTP, opcional)

  CG-01 Integración API → "base de datos": la alerta queda en events.json y su foto en disco
  CG-02 Integración admin.py ↔ hardware.py: cada tipo de evento manda el byte correcto al ESP32
  CG-03 Integración con servicio externo (n8n): payload correcto y tolerancia a la caída
  CG-04 Prueba de matriz: rol × acción (permisos)
  CG-05 Manejo de errores conociendo los códigos internos (400/401/403/413/503)
  CG-06 Regresión: la configuración guardada desde el panel cambia la respuesta de /predict
"""
import base64
import csv
import io
import json
from unittest import mock

import cv2
import numpy as np
import pytest

import admin
import data_store
import hardware

pytestmark = pytest.mark.caja_gris

from conftest import login  # noqa: E402  (tests/actividad/conftest.py)


def _foto_b64(color=120):
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), color, np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ── CG-01 ────────────────────────────────────────────────────────────────────

def test_CG01_alerta_por_api_queda_en_json_y_foto_en_disco(operator_client, tmp_data):
    cam = data_store.create_camera("Taller", "Sector B")
    foto = _foto_b64()
    with mock.patch("admin.enviar_comando_esp32"):
        r = operator_client.post("/admin/api/events", json={
            "type": "sin_casco", "confidence": 91.26, "camera": cam["id"], "persona": "  owen  ", "foto": foto})
    assert r.status_code == 200

    # Verificación directa en el almacenamiento (lo que el usuario no ve)
    [ev] = json.loads((tmp_data / "events.json").read_text(encoding="utf-8"))
    assert (ev["type"], ev["confidence"], ev["camera"], ev["ubicacion"], ev["persona"]) == \
        ("sin_casco", 91.3, "Taller", "Sector B", "owen")      # redondeo a 1 decimal y strip()
    ruta = tmp_data / "alertas" / f"{ev['id']}.webp"   # desde e0ac5f6 las fotos van en WebP
    assert ruta.exists()
    # La foto se re-codifica en el servidor: nunca se guardan los bytes del cliente tal cual
    assert ruta.read_bytes() != base64.b64decode(foto.split(",", 1)[1])
    assert cv2.imdecode(np.frombuffer(ruta.read_bytes(), np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)


# ── CG-02 ────────────────────────────────────────────────────────────────────

class _PuertoSerieFalso:
    """Reemplaza al ESP32: registra los bytes que le manda hardware.py."""

    def __init__(self):
        self.escritos = []
        self.is_open = True
        self.port = "COM-TEST"

    def write(self, b):
        self.escritos.append(b)

    def flush(self):
        pass

    def close(self):
        self.is_open = False


@pytest.fixture
def esp32(monkeypatch):
    puerto = _PuertoSerieFalso()
    monkeypatch.setattr(hardware, "esp32", puerto)
    monkeypatch.setattr(hardware, "_enabled", True)
    return puerto


@pytest.mark.parametrize("tipo, byte", [("sin_casco", b"R"), ("sin_chaleco", b"R"), ("con_casco", b"G")])
def test_CG02_evento_manda_el_comando_al_esp32(operator_client, esp32, tipo, byte):
    assert operator_client.post("/admin/api/events", json={"type": tipo, "confidence": 80}).status_code == 200
    assert esp32.escritos == [byte]


@pytest.mark.parametrize("estado, byte", [("sin_casco", b"R"), ("con_casco", b"G"), ("off", b"O")])
def test_CG02_api_alarm_llega_al_puerto_serie(operator_client, esp32, estado, byte):
    r = operator_client.post("/api/alarm", json={"state": estado})
    assert r.get_json() == {"ok": True, "cmd": byte.decode(), "sent_to_hardware": True}
    assert esp32.escritos == [byte]


def test_CG02_si_el_esp32_se_desconecta_el_sistema_sigue(operator_client, esp32):
    esp32.write = mock.MagicMock(side_effect=OSError("cable desenchufado"))
    r = operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80})
    assert r.status_code == 200 and len(data_store.load_events()) == 1
    assert hardware.get_status()["connected"] is False      # degradación controlada


# ── CG-03 ────────────────────────────────────────────────────────────────────

def test_CG03_webhook_n8n_recibe_la_alerta(operator_client, monkeypatch):
    data_store.create_camera("Taller", "Sector B")
    monkeypatch.setattr(admin, "N8N_WEBHOOK_URL", "https://n8n.invalid/webhook")
    with mock.patch("admin.requests.post") as post, mock.patch("admin.enviar_comando_esp32"):
        operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 88.4,
                                                        "camera": "Taller", "persona": "luciano"})
    payload = post.call_args.kwargs["json"]
    assert (payload["sensor"], payload["persona"], payload["ubicacion"], payload["alerta"]) == \
        ("Taller", "luciano", "Sector B", 88)
    assert post.call_args.kwargs["timeout"] == 3


def test_CG03_n8n_caido_no_rompe_el_registro(operator_client, monkeypatch):
    monkeypatch.setattr(admin, "N8N_WEBHOOK_URL", "https://n8n.invalid/webhook")
    with mock.patch("admin.requests.post", side_effect=ConnectionError("n8n caído")), \
         mock.patch("admin.enviar_comando_esp32"):
        r = operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80})
    assert r.status_code == 200 and len(data_store.load_events()) == 1


# ── CG-04  Prueba de matriz: rol × acción ────────────────────────────────────
# Supuesto (ver resumen_proyecto.md): "viewer" es un rol de solo lectura.
#   acción                      admin  operator  viewer
#   ver dashboard                 ✔       ✔        ✔
#   abrir el detector (/app)      ✔       ✔        ✔   (público, pero no registra)
#   abrir monitor multicámara     ✔       ✔        ✘
#   gestionar usuarios            ✔       ✘        ✘
#   borrar cámaras                ✔       ✘        ✘
#   registrar cámaras             ✔       ✔        ✘   ← DEF-01
#   registrar eventos / alarma    ✔       ✔        ✘   ← DEF-01

MATRIZ = [
    ("dashboard", {"admin": True, "operator": True, "viewer": True}),
    ("detector", {"admin": True, "operator": True, "viewer": True}),   # /app es público a propósito (demo); sin operador no registra nada
    ("monitor", {"admin": True, "operator": True, "viewer": False}),
    ("usuarios", {"admin": True, "operator": False, "viewer": False}),
    ("borrar_camara", {"admin": True, "operator": False, "viewer": False}),
]


def _puede(c, accion):
    if accion == "dashboard":
        r = c.get("/admin/")
        return r.status_code == 200
    if accion == "detector":
        return c.get("/app").status_code == 200
    if accion == "monitor":
        return c.get("/admin/monitor").status_code == 200
    if accion == "usuarios":
        return c.get("/admin/users").status_code == 200
    if accion == "borrar_camara":
        cam = data_store.create_camera("Borrable", "")
        c.post(f"/admin/cameras/{cam['id']}/delete")
        return all(x["id"] != cam["id"] for x in data_store.load_cameras())
    if accion == "registrar_camara":
        c.post("/admin/cameras", data={"name": "nueva"})
        return any(x["name"] == "nueva" for x in data_store.load_cameras())
    if accion == "registrar_evento":
        with mock.patch("admin.enviar_comando_esp32"):
            return c.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80}).status_code == 200
    raise ValueError(accion)


@pytest.mark.parametrize("rol", ["admin", "operator", "viewer"])
@pytest.mark.parametrize("accion, permitido", MATRIZ, ids=[m[0] for m in MATRIZ])
def test_CG04_matriz_de_permisos(flask_app, rol, accion, permitido):
    c = login(flask_app, f"u_{rol}", rol)
    assert _puede(c, accion) == permitido[rol]


@pytest.mark.defecto("DEF-01")
@pytest.mark.parametrize("accion", ["registrar_camara", "registrar_evento"])
def test_CG04_viewer_no_deberia_modificar_datos(flask_app, accion):
    c = login(flask_app, "u_viewer", "viewer")
    assert _puede(c, accion) is False


# ── CG-05  Manejo de errores con los códigos internos ───────────────────────

def test_CG05_codigos_de_error_de_la_api(client, operator_client, fake_model):
    assert client.post("/admin/api/events", json={}).status_code == 401           # sin sesión (API → JSON)
    assert client.get("/admin/").status_code == 302                                # sin sesión (página → login)
    assert operator_client.post("/admin/api/events", json={"type": "x"}).status_code == 400
    assert operator_client.post("/predict", json={"image": "sin-coma"}).status_code == 400
    grande = '{"image": "data:,' + "A" * (2 * 1024 * 1024) + '"}'
    assert operator_client.post("/predict", data=grande, content_type="application/json").status_code == 413


def test_CG05_sin_modelo_cargado_responde_503(flask_app, operator_client, monkeypatch):
    monkeypatch.setattr(flask_app, "model", None)
    monkeypatch.setattr(flask_app.helmet_detector, "disponible", lambda: False)
    r = operator_client.post("/predict", json={"image": _foto_b64()})
    assert r.status_code == 503 and "Modelo no cargado" in r.get_json()["error"]


@pytest.mark.defecto("DEF-04")
@pytest.mark.parametrize("url", ["/admin/audit?page=abc", "/admin/api/recent-events?limit=abc"])
def test_CG05_parametro_no_numerico_no_deberia_dar_500(operator_client, sin_propagar, url):
    assert operator_client.get(url).status_code < 500


@pytest.mark.defecto("DEF-08")
def test_CG05_imagen_base64_corrupta_deberia_dar_400(operator_client, fake_model):
    r = operator_client.post("/predict", json={"image": "data:image/jpeg;base64,@@@@"})
    assert r.status_code == 400


@pytest.mark.defecto("DEF-05")
def test_CG05_pagina_negativa_no_deberia_mostrar_eventos(tmp_data):
    for _ in range(30):
        data_store.append_event("con_casco", 90)
    eventos, _ = data_store.get_filtered_events(page=-1, per_page=25)
    assert eventos == []


# ── CG-06  Regresión e integración panel ↔ detector ─────────────────────────

def test_CG06_config_del_panel_cambia_la_respuesta_de_predict(flask_app, admin_c):
    form = {"action": "save", "min_confidence": 60, "confirm_threshold": 3, "scan_rate_ms": 100,
            "detectar_casco": "on"}                      # chaleco apagado desde el panel
    admin_c.post("/admin/config", data=form)
    cabeza = {"bbox": [1, 2, 3, 4], "clase": "casco", "conf": 0.9, "p_casco": 0.9, "p_sin": 0.1}
    with mock.patch.object(flask_app.helmet_detector, "disponible", return_value=True), \
         mock.patch.object(flask_app.helmet_detector, "detectar", return_value=[cabeza]):
        data = admin_c.post("/predict", json={"image": _foto_b64()}).get_json()
    assert data["modos"] == {"casco": True, "chaleco": False}
    assert "chaleco" not in data["faces"][0]


@pytest.mark.defecto("DEF-06")
def test_CG06_dashboard_cuenta_igual_alertas_y_grafico(tmp_data):
    data_store.append_event("sin_casco", 90)
    data_store.append_event("sin_chaleco", 90)
    s = data_store.get_stats()
    # Esperado: el gráfico por hora muestra todas las alertas que cuenta la tarjeta "Alertas 24 h"
    assert s["alerts_24h"] == sum(s["sin_casco_data"])


@pytest.mark.defecto("DEF-03")
def test_CG06_export_csv_neutraliza_formulas(operator_client):
    with mock.patch("admin.enviar_comando_esp32"):
        operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80,
                                                        "persona": '=HYPERLINK("http://x.invalid","ver")'})
    filas = list(csv.reader(io.StringIO(operator_client.get("/admin/audit/export").get_data(as_text=True))))
    persona = filas[1][5]
    # Esperado: la celda no empieza con = + - @ (si no, Excel la ejecuta como fórmula)
    assert not persona.startswith(("=", "+", "-", "@"))
