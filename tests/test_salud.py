"""Salud de las cámaras: lente tapada o desenfocada y señal perdida."""
import cv2
import numpy as np
import pytest

import aviso_telegram
import data_store
import motor
import salud


def _escena():
    """Imagen con bordes y tonos variados, como una obra."""
    rng = np.random.default_rng(3)
    img = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    cv2.rectangle(img, (100, 100), (300, 400), (20, 200, 240), -1)
    return img


def test_lente_bien_tapada_y_desenfocada():
    img = _escena()
    assert salud.lente(img) is None
    assert salud.lente(np.full((480, 640, 3), 5, np.uint8)) == "tapada"          # negra
    assert salud.lente(np.full((480, 640, 3), 140, np.uint8)) == "tapada"        # un solo tono (tela, cinta)
    assert salud.lente(cv2.GaussianBlur(img, (0, 0), 6)) == "desenfocada"


def test_vigia_avisa_una_vez_si_el_problema_dura():
    v = salud.Vigia(30)
    assert v.ver("camara_tapada", 0) is None
    assert v.ver("camara_tapada", 29) is None
    assert v.ver("camara_tapada", 30) == "camara_tapada"
    assert v.ver("camara_tapada", 90) is None          # ya avisó
    assert v.ver(None, 91) is None                     # se arregló: se rearma
    assert v.ver("camara_tapada", 100) is None and v.ver("camara_tapada", 131) == "camara_tapada"


def test_motor_registra_camara_tapada_con_foto(monkeypatch):
    cam = data_store.create_camera("Túnel", "", "usb", "0")
    eventos = []
    monkeypatch.setattr(motor, "registrar_evento", lambda *a: eventos.append(a))
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([], []))
    reloj = iter(range(0, 200, 2))
    monkeypatch.setattr(motor.time, "monotonic", lambda: next(reloj))
    hilo = motor.Camara(cam)
    for _ in range(20):                                # 40 s tapada
        hilo._procesar(np.full((480, 640, 3), 3, np.uint8))
    assert [e[0] for e in eventos] == ["camara_tapada"]
    assert eventos[0][4][:2] == b"\xff\xd8" and hilo.lente == "tapada"


def test_sin_senal_avisa_despues_de_un_minuto(monkeypatch):
    cam = data_store.create_camera("Rampa", "", "ip", "rtsp://10.0.0.9/stream")
    eventos = []
    monkeypatch.setattr(motor, "registrar_evento", lambda *a: eventos.append(a))
    hilo = motor.Camara(cam)
    for t in (0, 5, 15, 35):
        hilo._sin_senal(t)
    assert eventos == []
    hilo._sin_senal(75)
    hilo._sin_senal(135)
    assert eventos == [("camara_sin_senal", 100, cam["id"])]


def test_alerta_de_salud_no_hace_sonar_la_alarma_pero_avisa(flask_app, monkeypatch):
    import admin
    comandos, avisos = [], []
    monkeypatch.setattr(admin, "enviar_comando_esp32", comandos.append)
    monkeypatch.setattr(aviso_telegram, "avisar", lambda e, foto=None: avisos.append(e["type"]))
    cam = data_store.create_camera("Túnel", "Nivel -2")
    ev = admin.registrar_evento("camara_sin_senal", 100, cam["id"])
    assert ev["camera"] == "Túnel" and comandos == [] and avisos == ["camara_sin_senal"]
    assert "persona" not in ev


def test_telegram_leyenda_de_salud():
    t = aviso_telegram.leyenda({"type": "camara_tapada", "camera": "Túnel", "timestamp": "2026-11-22T03:10:00"})
    assert "TAPADA" in t and "Punto ciego desde las 03:10:00" in t


def test_nodo_informa_cpu_y_memoria():
    n = salud.nodo()
    assert set(n) == {"cpu", "memoria", "temperatura"}
    if salud.psutil:
        assert 0 <= n["memoria"] <= 100


# ── Zonas exentas y horario de exigencia ─────────────────────────────────────

from datetime import datetime  # noqa: E402


@pytest.mark.parametrize("horario, hora, exige", [
    (None, "03:00", True), ({"desde": "07:00", "hasta": "19:00"}, "06:59", False),
    ({"desde": "07:00", "hasta": "19:00"}, "07:00", True), ({"desde": "07:00", "hasta": "19:00"}, "19:00", False),
    ({"desde": "22:00", "hasta": "06:00"}, "23:30", True), ({"desde": "22:00", "hasta": "06:00"}, "12:00", False),
    ({"desde": "08:00", "hasta": "08:00"}, "12:00", True)])
def test_en_horario(horario, hora, exige):
    assert motor.en_horario(horario, datetime.strptime("2026-11-22 " + hora, "%Y-%m-%d %H:%M")) is exige


def _camara_con(monkeypatch, cara, **extra):
    cam = data_store.create_camera("Taller", "", "usb", "0")
    cam.update(extra)
    eventos = []
    monkeypatch.setattr(motor, "registrar_evento", lambda *a: eventos.append(a))
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([cara] if modos["casco"] else [], []))
    monkeypatch.setattr(motor.salud, "lente", lambda img: None)
    hilo = motor.Camara(cam)
    for _ in range(3):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    return [e[0] for e in eventos]


SIN_CASCO = {"bbox": [100, 100, 40, 40], "nombre": None, "con_casco": 5.0, "sin_casco": 95.0}


def test_sin_casco_en_zona_exenta_no_alerta(monkeypatch):
    assert _camara_con(monkeypatch, SIN_CASCO) == ["sin_casco"]
    exenta = [[[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]]          # la cabeza (120, 120) cae adentro
    assert _camara_con(monkeypatch, SIN_CASCO, exentas=exenta) == []


def test_fuera_de_horario_no_se_exige_casco(monkeypatch):
    ahora = datetime.now().strftime("%H:%M")
    fuera = {"desde": "00:00", "hasta": "00:01"} if ahora >= "00:01" else {"desde": "12:00", "hasta": "13:00"}
    assert _camara_con(monkeypatch, SIN_CASCO, horario=fuera) == []


def test_api_guarda_exentas_y_horario(operator_client):
    cam = data_store.create_camera("Taller", "", "usb", "0")
    url = f"/admin/api/cameras/{cam['id']}/zonas"
    poligono = [[0, 0], [0.5, 0], [0.5, 0.5]]
    r = operator_client.post(url, json={"zonas": [], "exentas": [poligono], "horario": {"desde": "07:00", "hasta": "19:00"}})
    assert r.status_code == 200
    [c] = data_store.load_cameras()
    assert c["exentas"] == [poligono] and c["horario"] == {"desde": "07:00", "hasta": "19:00"}
    operator_client.post(url, json={"zonas": []})                                  # sin exentas ni horario: no cambian
    assert data_store.load_cameras()[0]["horario"] == {"desde": "07:00", "hasta": "19:00"}
    operator_client.post(url, json={"zonas": [], "horario": None})                  # null: siempre
    assert data_store.load_cameras()[0]["horario"] is None
    assert operator_client.post(url, json={"zonas": [], "horario": {"desde": "7", "hasta": "19:00"}}).status_code == 400
    assert operator_client.post(url, json={"zonas": [], "exentas": "x"}).status_code == 400


def test_al_cerrar_el_horario_se_apaga_la_alarma(monkeypatch):
    cam = data_store.create_camera("Taller", "", "usb", "0")
    eventos = []
    monkeypatch.setattr(motor, "registrar_evento", lambda *a: eventos.append(a))
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([SIN_CASCO] if modos["casco"] else [], []))
    monkeypatch.setattr(motor.salud, "lente", lambda img: None)
    exige = [True]
    monkeypatch.setattr(motor, "en_horario", lambda h, ahora: exige[0])
    hilo = motor.Camara(cam)
    for _ in range(3):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert [e[0] for e in eventos] == ["sin_casco"] and hilo.alarma == "alerta"
    exige[0] = False                                    # 19:00: se termina el horario
    hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert [e[0] for e in eventos] == ["sin_casco", "con_casco"] and hilo.alarma is None


def test_escena_oscura_con_contraste_no_es_lente_tapada():
    noche = (np.random.default_rng(5).random((480, 640, 3)) * 40).astype(np.uint8)   # media ~20, con textura
    noche[200:260, 300:340] = 200                                                       # una luz
    assert salud.lente(noche) != "tapada"
