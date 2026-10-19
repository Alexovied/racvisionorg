"""Tests de los arreglos de seguridad (hallazgos #2, #3 y #6 de seguridad-rac)."""
import base64
from unittest import mock

import numpy as np
import pytest

import data_store

BIG = 2 * 1024 * 1024 + 1024  # apenas más de 2 MB


# ── Sin sesión: todo lo protegido responde 401 (API) o 302 (páginas) ─────────

@pytest.mark.parametrize("method,url", [
    ("post", "/predict"),
    ("post", "/api/alarm"),
    ("post", "/admin/api/events"),
    ("get", "/admin/api/config"),
    ("post", "/reset_buffer"),
])
def test_api_sin_sesion_devuelve_401(client, method, url):
    r = getattr(client, method)(url, json={})
    assert r.status_code == 401
    assert r.is_json and "error" in r.get_json()


def test_app_sin_sesion_redirige_al_login(client):
    r = client.get("/app")
    assert r.status_code == 302
    assert "/admin/login" in r.headers["Location"]
    assert "next=" in r.headers["Location"]


def test_landing_sigue_siendo_publica(client):
    with mock.patch("app.render_template", return_value="ok"):
        assert client.get("/").status_code == 200


def test_operador_puede_abrir_el_detector(operator_client):
    with mock.patch("app.render_template", return_value="detector"):
        assert operator_client.get("/app").status_code == 200


def test_alarma_sin_esp32_no_rompe(operator_client):
    r = operator_client.post("/api/alarm", json={"state": "off"})
    assert r.status_code == 200
    assert r.get_json()["sent_to_hardware"] is False


def test_login_no_redirige_a_sitios_externos(client):
    data_store.create_user("u1", "clave-u1", role="operator")
    r = client.post("/admin/login?next=//evil.example.com/",
                    data={"username": "u1", "password": "clave-u1"})
    assert r.status_code == 302
    assert "evil.example.com" not in r.headers["Location"]


# ── Validación de eventos (#3) ───────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    {"type": "unknown", "confidence": 80, "camera": "webcam"},
    {"type": "", "confidence": 80},
    {"confidence": 80},
    {"type": "sin_casco", "confidence": 150},
    {"type": "sin_casco", "confidence": -1},
    {"type": "sin_casco", "confidence": "80"},
    {"type": "sin_casco", "confidence": True},
    {"type": "sin_casco", "confidence": 80, "camera": "camara-inexistente"},
    {"type": "sin_casco", "confidence": 80, "camera": 123},
    ["no", "es", "objeto"],
])
def test_evento_invalido_devuelve_400(operator_client, payload):
    with mock.patch("admin.enviar_comando_esp32") as hw:
        r = operator_client.post("/admin/api/events", json=payload)
    assert r.status_code == 400
    hw.assert_not_called()
    assert data_store.load_events() == []


def test_evento_valido_webcam(operator_client):
    with mock.patch("admin.enviar_comando_esp32") as hw:
        r = operator_client.post("/admin/api/events",
                                 json={"type": "sin_casco", "confidence": 87.3, "camera": "webcam"})
    assert r.status_code == 200
    hw.assert_called_once_with("R")
    assert data_store.load_events()[0]["type"] == "sin_casco"


def test_evento_valido_camara_registrada(operator_client):
    cam = data_store.create_camera("Acceso Norte", "Portón 1")
    with mock.patch("admin.enviar_comando_esp32"):
        r1 = operator_client.post("/admin/api/events",
                                  json={"type": "con_casco", "confidence": 99, "camera": "Acceso Norte"})
        r2 = operator_client.post("/admin/api/events",
                                  json={"type": "con_casco", "confidence": 0, "camera": cam["id"]})
    assert r1.status_code == 200 and r2.status_code == 200


# ── Tamaño máximo de request (#6) ────────────────────────────────────────────

def test_predict_payload_mayor_a_2mb_devuelve_413(operator_client, fake_model):
    body = '{"image": "data:image/jpeg;base64,' + "A" * BIG + '"}'
    r = operator_client.post("/predict", data=body, content_type="application/json")
    assert r.status_code == 413
    fake_model.assert_not_called()


def test_eventos_payload_mayor_a_2mb_devuelve_413(operator_client):
    body = '{"type": "sin_casco", "confidence": 50, "camera": "' + "x" * BIG + '"}'
    r = operator_client.post("/admin/api/events", data=body, content_type="application/json")
    assert r.status_code == 413


# ── /predict: el preprocesamiento no cambió ──────────────────────────────────

def _frame_b64():
    import cv2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def test_predict_sin_cara(operator_client, fake_model):
    r = operator_client.post("/predict", json={"image": _frame_b64()})
    assert r.status_code == 200
    assert r.get_json() == {"has_face": False, "modos": {"casco": True, "chaleco": False}, "faces": []}


def test_predict_usa_rgb_128_y_preprocess_mobilenet(flask_app, operator_client, fake_model):
    crop = np.full((200, 150, 3), 255, dtype=np.uint8)
    flask_app.mobilenet_preprocess.reset_mock()
    with mock.patch.object(flask_app, "crop_head", return_value=[(crop, [1, 2, 3, 4])]):
        r = operator_client.post("/predict", json={"image": _frame_b64()})
    assert r.status_code == 200
    face = r.get_json()["faces"][0]
    assert face["bbox"] == [1, 2, 3, 4]
    assert face["con_casco"] == pytest.approx(90, abs=1e-3)

    arg = flask_app.mobilenet_preprocess.call_args[0][0]
    assert arg.shape == (128, 128, 3) and arg.dtype == np.float32
    batch = fake_model.call_args[0][0]
    assert batch.shape == (1, 128, 128, 3)


def test_predict_imagen_mal_formada_devuelve_400(operator_client, fake_model):
    assert operator_client.post("/predict", json={"image": 123}).status_code == 400
    assert operator_client.post("/predict", data="no-json", content_type="application/json").status_code == 400


def test_favicon_publico(client):
    r = client.get("/favicon.ico")
    assert r.status_code == 200 and len(r.data) > 0


def test_robots_y_sitemap_para_buscadores(client):
    robots = client.get("/robots.txt").get_data(as_text=True)
    assert "Disallow: /admin" in robots and "Sitemap: https://racvision.org/sitemap.xml" in robots
    assert "<loc>https://racvision.org/</loc>" in client.get("/sitemap.xml").get_data(as_text=True)
    landing = client.get("/").get_data(as_text=True)
    assert '<link rel="canonical" href="https://racvision.org/">' in landing and '"name": "RAC Vision"' in landing


def test_paginas_legales_publicas_y_enlazadas(client):
    for ruta in ("/privacidad", "/cookies", "/terminos"):
        r = client.get(ruta)
        assert r.status_code == 200 and "alex@racvision.org" in r.get_data(as_text=True)
    landing = client.get("/").get_data(as_text=True)
    assert all(f'href="/{p}"' in landing for p in ("privacidad", "cookies", "terminos"))


def test_landing_en_ingles_por_parametro_o_navegador(client):
    assert "No one goes in" in client.get("/?lang=en").get_data(as_text=True)
    assert "Nadie entra" in client.get("/?lang=es", headers={"Accept-Language": "en"}).get_data(as_text=True)
    assert 'lang="en"' in client.get("/", headers={"Accept-Language": "en-US"}).get_data(as_text=True)
    assert 'lang="es"' in client.get("/?lang=xx").get_data(as_text=True)


def test_panel_de_accesibilidad_en_todas_las_paginas(client, operator_client):
    for ruta in ("/", "/?lang=en", "/privacidad", "/cookies", "/terminos", "/admin/login"):
        assert "accesibilidad.js" in client.get(ruta).get_data(as_text=True), ruta
    assert "accesibilidad.js" in operator_client.get("/app").get_data(as_text=True)
