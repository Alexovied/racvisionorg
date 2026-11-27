"""Mejoras de app.py: tope del detector público, /healthz, cabeceras, proxy y carga perezosa de Keras."""
import threading

import pytest

import data_store


def test_detector_publico_con_tope_de_dos_a_la_vez(flask_app, client, operator_client, monkeypatch):
    monkeypatch.setattr(flask_app, "_ESPERA_ANONIMO", 0)
    assert flask_app._anonimos.acquire(blocking=False) and flask_app._anonimos.acquire(blocking=False)
    try:  # dos análisis anónimos en curso y se venció la espera: 429 (la página lo saltea, no es un fallo)
        assert client.post("/predict", json={"image": 123}).status_code == 429
        assert client.post("/api/demo/cara", json={}).status_code == 429
        assert client.delete("/api/demo/cara").status_code == 200           # borrar no analiza nada
        assert operator_client.post("/predict", json={"image": 123}).status_code == 400  # con sesión no hay tope
    finally:
        flask_app._anonimos.release()
        flask_app._anonimos.release()
    for _ in range(3):  # cada pedido devuelve su lugar al terminar
        assert client.post("/predict", json={"image": 123}).status_code == 400


def test_el_tercer_visitante_espera_su_lugar_en_vez_de_recibir_429(flask_app, client):
    assert flask_app._anonimos.acquire(blocking=False) and flask_app._anonimos.acquire(blocking=False)
    threading.Timer(0.2, flask_app._anonimos.release).start()   # uno de los dos análisis termina
    try:
        assert client.post("/predict", json={"image": 123}).status_code == 400
    finally:
        flask_app._anonimos.release()


def test_healthz_dice_que_detectores_cargaron(flask_app, client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "casco": flask_app.helmet_detector.disponible(),
                            "chaleco": flask_app.epp.disponible(), "pose": False}  # conftest apaga la pose


def test_cabeceras_de_seguridad(client):
    h = client.get("/").headers
    assert h["X-Frame-Options"] == "SAMEORIGIN" and h["Referrer-Policy"] == "same-origin"
    assert "max-age" in h["Strict-Transport-Security"]  # la cookie es Secure: se sirve por HTTPS


def test_con_proxy_el_bloqueo_de_login_es_por_ip_del_visitante(flask_app, client, monkeypatch):
    """Lo que app.py aplica con PROXIES=1: sin esto todos llegan con la IP del proxy."""
    original = flask_app.app.wsgi_app
    assert flask_app.con_proxies(original, "") is original and flask_app.con_proxies(original, "abc") is original
    monkeypatch.setattr(flask_app.app, "wsgi_app", flask_app.con_proxies(original, "1"))
    data_store.create_user("u3", "clave-u3", role="operator")
    atacante, admin = {"X-Forwarded-For": "1.1.1.1"}, {"X-Forwarded-For": "2.2.2.2"}
    for _ in range(5):
        client.post("/admin/login", data={"username": "u3", "password": "mala"}, headers=atacante)
    bien = {"username": "u3", "password": "clave-u3"}
    assert client.post("/admin/login", data=bien, headers=atacante).status_code == 429
    assert client.post("/admin/login", data=bien, headers=admin).status_code == 302


def test_keras_no_se_carga_si_esta_el_detector_yolo(flask_app):
    if flask_app.helmet_detector._net is None:
        pytest.skip("sin detector_casco.onnx la app sí carga el clasificador Keras")
    assert flask_app.model is None and flask_app.mobilenet_preprocess is None
