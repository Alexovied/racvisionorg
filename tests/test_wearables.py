"""Wearables: registro con token, telemetría, reglas de alerta, cruce con la cámara y aprendizaje."""
from datetime import datetime, timedelta

import pytest

import admin
import aprendizaje
import data_store
import wearables


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    wearables._estados.clear()
    monkeypatch.setattr(admin, "_iniciar_vigilancia", lambda: None)   # sin hilos en los tests
    yield
    wearables._estados.clear()


@pytest.fixture
def admin_client(flask_app):
    c = flask_app.app.test_client()
    r = c.post("/admin/login", data={"username": "admin", "password": "admin-solo-para-tests"})
    assert r.status_code == 302
    return c


def _mandar(client, token, **lectura):
    return client.post("/admin/api/dispositivos/telemetria", json=lectura,
                       headers={"Authorization": f"Bearer {token}"})


# ── Reglas (sin servidor) ────────────────────────────────────────────────────

def test_lectura_normal_no_avisa():
    est = wearables.Estado()
    for t in range(0, 120, 5):
        assert wearables.evaluar(est, {"bpm": 85, "spo2": 97, "actividad": 0.4}, t) == []


def test_sos_avisa_una_vez_y_se_rearma():
    est = wearables.Estado()
    assert wearables.evaluar(est, {"sos": True}, 0) == [("boton_sos", "botón de pánico")]
    assert wearables.evaluar(est, {"sos": True}, 5) == []          # sigue apretado: no repite
    assert wearables.evaluar(est, {"sos": False}, 10) == []
    assert [t for t, _ in wearables.evaluar(est, {"sos": True}, 15)] == ["boton_sos"]


def test_hombre_muerto_del_equipo():
    est = wearables.Estado()
    assert [t for t, _ in wearables.evaluar(est, {"hombre_muerto": True}, 0)] == ["hombre_muerto"]


def test_taquicardia_solo_si_es_sostenida():
    est = wearables.Estado()
    assert wearables.evaluar(est, {"bpm": 175, "actividad": 0.5}, 0) == []
    assert wearables.evaluar(est, {"bpm": 90, "actividad": 0.5}, 10) == []     # se cortó: vuelve a contar
    assert wearables.evaluar(est, {"bpm": 175, "actividad": 0.5}, 15) == []
    assert wearables.evaluar(est, {"bpm": 176, "actividad": 0.5}, 30) == []
    assert wearables.evaluar(est, {"bpm": 178, "actividad": 0.5}, 35) == [("ritmo_cardiaco", "taquicardia 178 lpm")]


def test_sin_pulso_con_sensor_apoyado_pero_no_si_se_lo_saco():
    est = wearables.Estado()
    wearables.evaluar(est, {"bpm": 0, "contacto": False, "actividad": 0.3}, 0)
    assert wearables.evaluar(est, {"bpm": 0, "contacto": False, "actividad": 0.3}, 60) == []
    est = wearables.Estado()
    wearables.evaluar(est, {"bpm": 0, "contacto": True, "actividad": 0.3}, 0)
    assert wearables.evaluar(est, {"bpm": 0, "contacto": True, "actividad": 0.3}, 25) == [("ritmo_cardiaco", "sin pulso")]


def test_desmayo_caida_y_quieto():
    est = wearables.Estado()
    assert wearables.evaluar(est, {"bpm": 95, "caida": True, "actividad": 0.01}, 0) == []
    assert wearables.evaluar(est, {"bpm": 90, "actividad": 0.01}, 20) == []
    assert wearables.evaluar(est, {"bpm": 88, "actividad": 0.01}, 31) == [("posible_desmayo", "quieto después de una caída")]


def test_caida_y_se_levanta_no_es_desmayo():
    est = wearables.Estado()
    wearables.evaluar(est, {"bpm": 95, "caida": True, "actividad": 0.01}, 0)
    wearables.evaluar(est, {"bpm": 100, "actividad": 0.6}, 10)    # se movió
    assert wearables.evaluar(est, {"bpm": 90, "actividad": 0.01}, 35) == []


def test_quieto_sin_caida_y_pulso_normal_no_avisa():
    """Un operador sentado en la cabina no es un desmayo."""
    est = wearables.Estado()
    for t in range(0, 300, 5):
        assert wearables.evaluar(est, {"bpm": 70, "actividad": 0.01}, t) == []


def test_umbrales_propios_del_trabajador():
    est = wearables.Estado()
    wearables.evaluar(est, {"bpm": 175}, 0, {"bpm_max": 185})
    assert wearables.evaluar(est, {"bpm": 175}, 60, {"bpm_max": 185}) == []


def test_equipo_callado_avisa_una_vez():
    estados = {"a": wearables.Estado(), "b": wearables.Estado()}
    wearables.evaluar(estados["a"], {"bpm": 80}, 0)
    wearables.evaluar(estados["b"], {"bpm": 80}, 80)
    assert wearables.callados(estados, 100) == ["a"]
    assert wearables.callados(estados, 200) == ["b"]     # "a" ya avisó
    wearables.evaluar(estados["a"], {"bpm": 80}, 210)     # volvió
    assert wearables.callados(estados, 400) == ["a"]


@pytest.mark.parametrize("dato", [{"bpm": "80"}, {"bpm": 400}, {"sos": "si"}, {}, [], {"spo2": float("nan")}])
def test_lecturas_invalidas(dato):
    lectura, error = wearables.validar_lectura(dato)
    assert lectura is None and error


# ── API ──────────────────────────────────────────────────────────────────────

def test_telemetria_sin_token_o_con_token_malo(client):
    assert client.post("/admin/api/dispositivos/telemetria", json={"bpm": 80}).status_code == 401
    assert _mandar(client, "rac_inventado", bpm=80).status_code == 401


def test_telemetria_no_pide_csrf_pero_el_resto_si(flask_app, monkeypatch):
    """El equipo no tiene sesión: el CSRF no aplica a su endpoint, sí al resto del panel."""
    monkeypatch.setitem(flask_app.app.config, "WTF_CSRF_ENABLED", True)
    _, token = wearables.crear("Pulsera 01", "Juan")
    c = flask_app.app.test_client()
    assert _mandar(c, token, bpm=80).status_code == 200
    assert c.post("/admin/api/events/x/feedback", json={"veredicto": "real"}).status_code == 403


def test_sos_por_api_queda_como_alerta(client, tmp_data):
    disp, token = wearables.crear("Pulsera 01", "Juan Pérez")
    r = _mandar(client, token, bpm=110, sos=True, bateria=76)
    assert r.status_code == 200 and r.json["alertas"] == ["boton_sos"] and r.json["aviso_recibido"]
    ev = data_store.load_events()[0]
    assert (ev["type"], ev["persona"], ev["camera"], ev["dispositivo"]) == ("boton_sos", "Juan Pérez", "Pulsera 01", disp["id"])
    assert ev["medicion"] == {"bpm": 110, "bateria": 76}
    assert _mandar(client, token, bpm=110, sos=True).json["alertas"] == []   # no repite


def test_token_se_guarda_hasheado_y_se_puede_regenerar(client, tmp_data):
    disp, token = wearables.crear("Pulsera 01", "Juan")
    assert token not in (tmp_data / "dispositivos.json").read_text()
    nuevo = wearables.regenerar_token(disp["id"])
    assert _mandar(client, token, bpm=80).status_code == 401
    assert _mandar(client, nuevo, bpm=80).status_code == 200


def test_panel_registra_y_muestra_el_token_una_vez(admin_client, operator_client):
    r = admin_client.post("/admin/wearables", data={"nombre": "Pulsera 07", "persona": "Ana"})
    assert r.status_code == 200 and b"rac_" in r.data
    assert b"rac_" not in admin_client.get("/admin/wearables").data
    lista = operator_client.get("/admin/api/wearables").json
    assert [(w["nombre"], w["persona"], w["situacion"]) for w in lista] == [("Pulsera 07", "Ana", "sin_datos")]
    # el operador no registra equipos
    operator_client.post("/admin/wearables", data={"nombre": "X", "persona": "Y"})
    assert len(wearables.cargar()) == 1


def test_estado_en_vivo(client, operator_client):
    _, token = wearables.crear("Pulsera 01", "Juan")
    _mandar(client, token, bpm=72, spo2=98, bateria=90, actividad=0.3)
    w = operator_client.get("/admin/api/wearables").json[0]
    assert w["situacion"] == "ok" and w["lectura"]["bpm"] == 72
    _mandar(client, token, sos=True)
    assert operator_client.get("/admin/api/wearables").json[0]["situacion"] == "alerta"


def test_sin_senal_genera_alerta_pero_no_suena_la_sirena(client, monkeypatch):
    sirena = []
    monkeypatch.setattr(admin, "enviar_comando_esp32", sirena.append)
    disp, token = wearables.crear("Pulsera 01", "Juan")
    _mandar(client, token, bpm=80)
    ultima = wearables._estados[disp["id"]].ultima[0]
    eventos = admin.revisar_wearables(ultima + wearables.SIN_SENAL_S + 1)
    assert [e["type"] for e in eventos] == ["wearable_sin_senal"] and sirena == []
    assert admin.revisar_wearables(ultima + 500) == []


# ── Cruce con la cámara ─────────────────────────────────────────────────────

def test_desmayo_confirmado_por_la_camara(tmp_data):
    cam = data_store.create_camera("Galería 3", "Nivel -120", "ip")
    caida = admin.registrar_evento("hombre_caido", 88, cam["id"])
    disp, _ = wearables.crear("Pulsera 01", "Juan", cam["id"])
    ev = admin.alerta_de_wearable(wearables.obtener(disp["id"]), "posible_desmayo", "quieto después de una caída", {})
    assert ev["confirmado_por"] == "cámara Galería 3" and ev["evento_relacionado"] == caida["id"]
    assert ev["confidence"] == 98 and ev["ubicacion"] == "Nivel -120"


def test_caida_de_otra_camara_no_confirma(tmp_data):
    cam1 = data_store.create_camera("Galería 3", "", "ip")
    cam2 = data_store.create_camera("Taller", "", "ip")
    admin.registrar_evento("hombre_caido", 88, cam2["id"])
    disp, _ = wearables.crear("Pulsera 01", "Juan", cam1["id"])
    ev = admin.alerta_de_wearable(wearables.obtener(disp["id"]), "posible_desmayo", "", {})
    assert "confirmado_por" not in ev


def test_camara_confirma_un_sos_reciente(tmp_data):
    disp, _ = wearables.crear("Pulsera 01", "Juan")
    sos = admin.alerta_de_wearable(wearables.obtener(disp["id"]), "boton_sos", "botón de pánico", {})
    caida = admin.registrar_evento("hombre_caido", 80, "webcam")
    assert caida["evento_relacionado"] == sos["id"] and caida["confirmado_por"].startswith("wearable Pulsera 01")


def test_buscar_confirmacion_respeta_la_ventana():
    ahora = datetime(2026, 11, 22, 12, 0, 0)
    eventos = [{"id": "1", "type": "hombre_caido", "camera": "A", "timestamp": (ahora - timedelta(minutes=5)).isoformat()}]
    assert wearables.buscar_confirmacion(eventos, ("hombre_caido",), ahora) is None
    assert wearables.buscar_confirmacion(eventos, ("hombre_caido",), ahora, ventana_s=600)["id"] == "1"


# ── Feedback y aprendizaje ──────────────────────────────────────────────────

def test_aviso_urgente_hasta_que_lo_marcan(client, operator_client):
    _, token = wearables.crear("Pulsera 01", "Juan")
    _mandar(client, token, sos=True)
    urgentes = operator_client.get("/admin/api/alertas-urgentes").json
    assert [u["type"] for u in urgentes] == ["boton_sos"] and urgentes[0]["persona"] == "Juan"
    r = operator_client.post(f"/admin/api/events/{urgentes[0]['id']}/feedback", json={"veredicto": "real"})
    assert r.status_code == 200
    assert operator_client.get("/admin/api/alertas-urgentes").json == []
    assert data_store.get_event(urgentes[0]["id"])["feedback"]["usuario"] == "operador_test"


def test_feedback_valida(operator_client):
    ev = data_store.append_event("con_casco", 90, "webcam")
    assert operator_client.post(f"/admin/api/events/{ev['id']}/feedback", json={"veredicto": "falsa"}).status_code == 404
    assert operator_client.post("/admin/api/events/nada/feedback", json={"veredicto": "real"}).status_code == 404
    ev = data_store.append_event("sin_casco", 90, "webcam")
    assert operator_client.post(f"/admin/api/events/{ev['id']}/feedback", json={"veredicto": "quizas"}).status_code == 400


def test_viewer_no_marca(flask_app):
    data_store.create_user("mira", "clave-mira", role="viewer")
    c = flask_app.app.test_client()
    c.post("/admin/login", data={"username": "mira", "password": "clave-mira"})
    ev = data_store.append_event("sin_casco", 90, "webcam")
    assert c.post(f"/admin/api/events/{ev['id']}/feedback", json={"veredicto": "real"}).status_code == 403


def test_falsa_taquicardia_sube_el_umbral_de_ese_trabajador(client, operator_client):
    disp, token = wearables.crear("Pulsera 01", "Minero que corre")
    otro, _ = wearables.crear("Pulsera 02", "Otro")
    ev = admin.alerta_de_wearable(disp, "ritmo_cardiaco", "taquicardia 172 lpm", {"bpm": 172})
    r = operator_client.post(f"/admin/api/events/{ev['id']}/feedback", json={"veredicto": "falsa"})
    assert "177" in r.json["aprendido"]
    assert wearables.umbrales_de(wearables.obtener(disp["id"]))["bpm_max"] == 177
    assert wearables.umbrales_de(wearables.obtener(otro["id"]))["bpm_max"] == 160
    # con 172 sostenido ya no avisa
    est = wearables.Estado()
    u = wearables.umbrales_de(wearables.obtener(disp["id"]))
    wearables.evaluar(est, {"bpm": 172}, 0, u)
    assert wearables.evaluar(est, {"bpm": 172}, 60, u) == []


def test_aprendizaje_tiene_topes():
    disp, _ = wearables.crear("Pulsera 01", "Juan")
    for _ in range(10):
        wearables.aprender_de_falsa({"id": "x", "dispositivo": disp["id"], "type": "ritmo_cardiaco",
                                     "detalle": "taquicardia 230 lpm", "medicion": {"bpm": 230}})
        wearables.aprender_de_falsa({"id": "y", "dispositivo": disp["id"], "type": "posible_desmayo", "medicion": {}})
    u = wearables.umbrales_de(wearables.obtener(disp["id"]))
    assert u["bpm_max"] == wearables.TOPES["bpm_max"] and u["inactivo_s"] == wearables.TOPES["inactivo_s"]
    assert wearables.restablecer_umbrales(disp["id"])
    assert wearables.umbrales_de(wearables.obtener(disp["id"])) == wearables.UMBRALES


def test_sugerencia_de_confianza_para_camaras():
    eventos = []
    for i, (conf, veredicto) in enumerate([(55, "falsa"), (58, "falsa"), (60, "falsa"), (62, "falsa"), (66, "real"),
                                           (70, "real"), (75, "real"), (80, "real"), (85, "real"), (64, "falsa")]):
        eventos.append({"id": str(i), "type": "sin_casco", "confidence": conf, "feedback": {"veredicto": veredicto}})
    s = aprendizaje.sugerir_confianza(eventos, actual=51)
    assert s == {"confianza": 66, "falsas_evitadas": 5, "falsas": 5, "reales": 5}
    assert aprendizaje.sugerir_confianza(eventos, actual=70) is None
    assert aprendizaje.sugerir_confianza(eventos[:5], actual=51) is None    # pocas marcas
    assert aprendizaje.precision(eventos)["sin_casco"] == {"reales": 5, "falsas": 5, "precision": 50}


def test_auditoria_muestra_lo_aprendido(operator_client):
    ev = data_store.append_event("sin_casco", 90, "webcam")
    data_store.set_feedback(ev["id"], "real", "operador_test")
    html = operator_client.get("/admin/audit").get_data(as_text=True)
    assert "Lo que aprendió el sistema" in html and "100%" in html


def test_exportar_fotos_marcadas(tmp_path, tmp_data):
    import cv2
    import numpy as np
    ok, jpg = cv2.imencode(".jpg", np.full((32, 32, 3), 100, np.uint8))
    ev = admin.registrar_evento("sin_casco", 70, "webcam", None, jpg.tobytes())
    data_store.set_feedback(ev["id"], "falsa", "op")
    assert aprendizaje.exportar_fotos(data_store.load_events(), str(tmp_path / "ds")) == 1
    assert (tmp_path / "ds" / "sin_casco" / "falsa" / f"{ev['id']}.webp").exists()


# ── Simulador ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("escenario", ["normal", "sos", "hombre_muerto", "desmayo", "taquicardia", "sin_pulso"])
def test_simulador_dispara_lo_que_dice(escenario):
    """Las lecturas del simulador pasan la validación y, con el reloj del escenario, disparan su alerta."""
    import simulador_wearable as sim
    esperado = {"normal": set(), "sos": {"boton_sos"}, "hombre_muerto": {"hombre_muerto"},
                "desmayo": {"posible_desmayo"}, "taquicardia": {"ritmo_cardiaco"}, "sin_pulso": {"ritmo_cardiaco"}}
    est, vistas = wearables.Estado(), set()
    for t in range(0, 60, 2):
        lectura, error = wearables.validar_lectura(sim.lectura(escenario, t, 80))
        assert error is None
        vistas |= {tipo for tipo, _ in wearables.evaluar(est, lectura, t)}
    assert esperado[escenario] <= vistas
    if escenario == "normal":
        assert vistas == set()
