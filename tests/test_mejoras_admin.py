"""Tests de los arreglos de la auditoría sobre admin.py (login, sesiones, cámaras, feedback, errores 500)."""
import re
import threading
import time

import cv2
import numpy as np

import admin
import data_store
import wearables


def test_login_fallido_no_guarda_nombres_enormes(client):
    client.post("/admin/login", data={"username": "x" * 10000, "password": "mala"})
    assert admin._intentos and all(len(usuario) <= 80 for _, usuario in admin._intentos)


def test_cambiar_la_clave_cierra_las_sesiones_abiertas(operator_client):
    assert operator_client.get("/admin/").status_code == 200
    usuario = data_store.get_user_by_username("operador_test")
    data_store.update_user(usuario["id"], password="clave-nueva")
    assert operator_client.get("/admin/").status_code == 302          # la cookie vieja ya no sirve
    with operator_client.session_transaction() as s:
        assert usuario["password_hash"][-16:] not in str(dict(s))     # la cookie no lleva el hash


def test_admin_que_cambia_su_propia_clave_sigue_adentro(flask_app):
    jefe = data_store.create_user("jefe", "clave-jefe", role="admin")
    c = flask_app.app.test_client()
    c.post("/admin/login", data={"username": "jefe", "password": "clave-jefe"})
    c.post("/admin/users", data={"action": "edit", "user_id": jefe["id"], "password": "clave-nueva"})
    assert c.get("/admin/users").status_code == 200


def test_la_pagina_de_camaras_no_muestra_la_clave_del_rtsp(flask_app):
    data_store.create_camera("Rampa", "", "ip", "rtsp://admin:S3creta@10.0.0.5/s")
    data_store.create_camera("Patio", "", "ip", "http://10.0.0.6/v.cgi?user=admin&pwd=Otr4")
    data_store.create_user("mira", "clave-mira", role="viewer")
    c = flask_app.app.test_client()
    c.post("/admin/login", data={"username": "mira", "password": "clave-mira"})
    html = c.get("/admin/cameras").get_data(as_text=True)
    assert "S3creta" not in html and "Otr4" not in html
    assert "rtsp://***@10.0.0.5/s" in html                            # la dirección se sigue viendo


def test_marcar_dos_veces_la_misma_falsa_afloja_una_sola_vez(operator_client):
    disp, _ = wearables.crear("Pulsera 01", "Minero")
    antes = wearables.umbrales_de(wearables.obtener(disp["id"]))["inactivo_s"]
    ev = admin.alerta_de_wearable(disp, "posible_desmayo", "sin movimiento", {})
    for _ in range(2):
        r = operator_client.post(f"/admin/api/events/{ev['id']}/feedback", json={"veredicto": "falsa"})
        assert r.status_code == 200
    assert r.json["aprendido"] is None
    assert wearables.umbrales_de(wearables.obtener(disp["id"]))["inactivo_s"] == antes + 15


def test_dos_operadores_a_la_vez_con_la_misma_falsa_aflojan_una_sola_vez(flask_app, operator_client, monkeypatch):
    disp, _ = wearables.crear("Pulsera 02", "Minero")
    antes = wearables.umbrales_de(wearables.obtener(disp["id"]))["inactivo_s"]
    ev = admin.alerta_de_wearable(disp, "posible_desmayo", "sin movimiento", {})
    otro = flask_app.app.test_client()
    otro.post("/admin/login", data={"username": "operador_test", "password": "clave-operador"})
    leer = data_store.get_event

    def lento(event_id):   # events.json grande: los dos leerían antes de que alguno guarde
        e = leer(event_id)
        time.sleep(0.2)
        return e
    monkeypatch.setattr(data_store, "get_event", lento)
    url = f"/admin/api/events/{ev['id']}/feedback"
    hilos = [threading.Thread(target=c.post, args=(url,), kwargs={"json": {"veredicto": "falsa"}})
             for c in (operator_client, otro)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert wearables.umbrales_de(wearables.obtener(disp["id"]))["inactivo_s"] == antes + 15


def test_relay_con_cuerpo_vacio_o_foto_gigante_da_400(operator_client, client, monkeypatch):
    r = operator_client.post("/admin/cameras", data={"name": "Oryx", "type": "relay"}, follow_redirects=True)
    token = re.search(r"--token (rac_\S+)", r.get_data(as_text=True)).group(1)
    url = f"/admin/api/cameras/{data_store.load_cameras()[0]['id']}/cuadro"
    cab = {"Authorization": f"Bearer {token}"}
    assert client.post(url, data=b"", headers=cab).status_code == 400

    def gigante(*a):
        raise cv2.error("mas pixeles que OPENCV_IO_MAX_IMAGE_PIXELS")
    monkeypatch.setattr(admin.cv2, "imdecode", gigante)
    assert client.post(url, data=b"jpeg", headers=cab).status_code == 400
    assert admin.a_webp(b"jpeg") is None and admin.anonimizar(b"jpeg") is None


def test_error_al_registrar_evento_no_muestra_detalles_y_queda_en_el_log(operator_client, monkeypatch, capsys):
    def falla(*a, **k):
        raise PermissionError("C:/ruta/interna/data/events.json")
    monkeypatch.setattr(admin, "registrar_evento", falla)
    r = operator_client.post("/admin/api/events", json={"type": "con_casco", "confidence": 90})
    assert r.status_code == 500 and "ruta" not in r.get_data(as_text=True)
    assert "PermissionError" in capsys.readouterr().err


def test_n8n_caido_no_deja_la_url_del_webhook_en_el_log(monkeypatch, capsys):
    monkeypatch.setattr(admin, "N8N_WEBHOOK_URL", "https://n8n.example/webhook/SECRETO-123")

    def caido(url, **k):
        raise admin.requests.ConnectionError(f"Max retries exceeded with url: {url}")
    monkeypatch.setattr(admin.requests, "post", caido)
    admin._avisar_n8n({"tipo_riesgo": "sin_casco"}, {})
    salida = capsys.readouterr().out
    assert "ConnectionError" in salida and "SECRETO" not in salida
