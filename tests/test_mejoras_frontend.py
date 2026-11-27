"""Mejoras de las plantillas: nombres dentro de confirm() y avisos de conexión del monitor.
El JS no corre acá (no hay navegador): se fija el markup que llega a la página."""
import html
import re

import pytest

import data_store
import face_id


@pytest.fixture
def admin_client(flask_app):
    data_store.create_user("admin_test", "clave-admin", role="admin")
    c = flask_app.app.test_client()
    assert c.post("/admin/login", data={"username": "admin_test", "password": "clave-admin"}).status_code == 302
    return c


def _manejadores(pagina):
    # Todos los on...= (onsubmit, onclick...). Como lo ve el navegador: el atributo se decodifica antes de compilarse como JS
    return [html.unescape(a) for a in re.findall(r'\bon\w+="([^"]*)"', pagina)]


def test_nombre_con_comillas_no_entra_al_js_de_confirmar(admin_client, monkeypatch):
    malo = "'+alert(1)+'"
    data_store.create_camera(malo, "", "webcam")
    data_store.create_user("o'neil" + malo, "clave-1234", role="operator")
    monkeypatch.setattr(face_id, "listar", lambda: [{"nombre": "O'Neil" + malo, "fotos": 3, "alta": "", "por": ""}])
    for ruta in ("/admin/cameras", "/admin/users", "/admin/personas"):
        pagina = admin_client.get(ruta).get_data(as_text=True)
        handlers = _manejadores(pagina)
        assert "return confirm(this.dataset.confirmar)" in handlers, ruta
        assert not [h for h in handlers if "alert(1)" in h], ruta
        # el texto del cartel viaja en un atributo de datos (escapado como HTML), no dentro del JS
        assert any(malo in html.unescape(t) for t in re.findall(r'data-confirmar="([^"]*)"', pagina)), ruta


def test_monitor_avisa_si_el_servidor_deja_de_responder(operator_client):
    data_store.create_camera("Entrada", "", "webcam")
    pagina = operator_client.get("/admin/monitor").get_data(as_text=True)
    assert "Sin conexión con el servidor" in pagina     # cámaras del servidor: sondeo de /admin/api/motor
    assert "'SIN CONEXIÓN', 'El servidor no responde'" in pagina   # webcam: /predict
    # evento o alarma rechazados por sesión vencida: se vuelve al login
    assert pagina.count(".then(sesionVencida)") + pagina.count("sesionVencida(r)") >= 4   # también /predict (403 de CSRF)
    # un reintento viejo no pisa un estado más nuevo de la cámara
    assert "if (vigente()) enviarEvento(evento, intento + 1)" in pagina


def test_aviso_de_urgentes_no_se_rearma_si_no_cambio(operator_client):
    pagina = operator_client.get("/admin/monitor").get_data(as_text=True)
    assert "if (firma === ultima) return;" in pagina
    assert "No se pudo marcar la alerta" in pagina
