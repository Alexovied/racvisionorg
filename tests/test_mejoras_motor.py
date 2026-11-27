"""Arreglos de la auditoría del motor: rearme de alertas, alarma física, hombre caído y señal."""
import numpy as np
import pytest

import data_store
import motor

NEGRO = lambda: np.zeros((480, 640, 3), np.uint8)
EN_ZONA = [100, 100, 80, 300]     # apoya en x=140: dentro de la zona de la fixture


@pytest.fixture
def camara(monkeypatch):
    """Cámara con una zona en la mitad izquierda; analizar() lo simula cada test."""
    cam = data_store.create_camera("Portón", "Sector A", "usb", "0")
    cam = data_store.set_zonas(cam["id"], [[[0, 0], [0.5, 0], [0.5, 1], [0, 1]]])
    eventos = []
    monkeypatch.setattr(motor, "registrar_evento", lambda *a: eventos.append(a))
    monkeypatch.setattr(motor.salud, "lente", lambda img: None)
    return motor.Camara(cam), eventos


def _simular(monkeypatch, caras=(), personas=()):
    monkeypatch.setattr(motor, "analizar",
                        lambda img, modos, con_personas=False, **kw: (list(caras), list(personas)))


def _esqueleto(hombros, caderas):
    p = np.zeros((17, 3), np.float32)
    for i, xy in zip((5, 6, 11, 12), hombros + caderas):
        p[i] = (*xy, 0.9)
    return p


# ── Seguimiento ──────────────────────────────────────────────────────────────

def test_otro_infractor_avisa_despues_de_la_camara_vacia():
    s = motor.Seguimiento()
    assert [s.procesar({"casco": "falta"}, False, 3, 10, 0.0) for _ in range(3)] == [[], [], ["casco"]]
    for _ in range(motor.VACIOS_RESET):                 # se le pierde la cabeza unos segundos: no repite
        s.procesar({"casco": None}, False, 3, 10, 5.0)
    assert [s.procesar({"casco": "falta"}, False, 3, 10, 6.0) for _ in range(3)] == [[], [], []]
    for _ in range(motor.VACIOS_RESET):                 # la cámara queda vacía más de REARME_S
        s.procesar({"casco": None}, False, 3, 10, 100.0)
    assert s.resumen(["casco"]) == ("alerta", "SIN CASCO")      # la alerta sigue a la vista
    assert [s.procesar({"casco": "falta"}, False, 3, 10, 101.0) for _ in range(3)] == [[], [], ["casco"]]
    assert s.procesar({"casco": "falta"}, False, 3, 10, 101.0) == []


def test_votos_sueltos_vencen_con_la_camara_vacia():
    s = motor.Seguimiento()
    assert [s.procesar({"casco": "falta"}, False, 3, 10, 0.0) for _ in range(2)] == [[], []]
    for _ in range(motor.VACIOS_RESET):
        s.procesar({"casco": None}, False, 3, 10, 3600.0)
    assert s.procesar({"casco": "falta"}, False, 3, 10, 3601.0) == []   # un frame malo solo no alcanza


def test_hombre_caido_tolera_huecos_cortos_entre_cuadros():
    s = motor.Seguimiento()   # relay con microcortes: un cuadro cada 3 s, caida_segundos = 10
    assert [s.procesar({}, True, 3, 10, 100 + 3 * i) for i in range(5)] == [[]] * 4 + [["caida"]]


# ── analizar ─────────────────────────────────────────────────────────────────

def test_sin_cuerpos_no_se_corre_el_detector_de_cascos(monkeypatch):
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    monkeypatch.setattr(motor.pose, "detectar", lambda img: [])
    llamadas = []
    monkeypatch.setattr(motor.helmet_detector, "detectar", lambda img: llamadas.append(1) or [])
    assert motor.analizar(NEGRO(), {"casco": True, "chaleco": False}) == ([], [])
    assert llamadas == []


def test_companero_agachado_no_anula_la_caida(monkeypatch):
    caja = [90, 280, 220, 70]
    caido = (caja, _esqueleto([(100, 290), (100, 310)], [(300, 325), (300, 345)]))               # torso a 80°
    ayudante = ([400, 180, 120, 140], _esqueleto([(416, 190), (416, 210)], [(500, 290), (500, 310)]))  # a 40°
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    monkeypatch.setattr(motor.pose, "detectar", lambda img: [caido, ayudante])
    monkeypatch.setattr(motor.cuerpo, "es_persona", lambda *a: True)
    monkeypatch.setattr(motor.face_id, "identificar", lambda *a, **k: None)
    monkeypatch.setattr(motor.helmet_detector, "detectar", lambda img: [
        {"bbox": [430, 150, 40, 40], "clase": "casco", "conf": 0.9, "p_casco": 0.9, "p_sin": 0.1}])
    modos, info = {"casco": True, "chaleco": False}, {}
    caras, personas = motor.analizar(NEGRO(), modos, info=info)
    assert len(caras) == 1 and info["inclinacion"] == 0     # no se enderezó la imagen
    assert motor.acostados(NEGRO(), personas, caras, modos, info["cuerpos"], info["inclinacion"]) == [caja]


# ── Cámara ───────────────────────────────────────────────────────────────────

def test_fin_de_horario_no_apaga_la_alarma_con_alguien_en_zona(camara, monkeypatch):
    hilo, eventos = camara
    _simular(monkeypatch, personas=[EN_ZONA])
    exige = [True]
    monkeypatch.setattr(motor, "en_horario", lambda h, ahora: exige[0])
    for _ in range(3):
        hilo._procesar(NEGRO())
    exige[0] = False                                    # termina el horario con la persona adentro
    for _ in range(5):
        hilo._procesar(NEGRO())
    assert [e[0] for e in eventos] == ["zona_prohibida"] and hilo.alarma == "alerta"


def _reloj(monkeypatch, paso):
    t = [100.0]
    monkeypatch.setattr(motor.time, "monotonic", lambda: t.__setitem__(0, t[0] + paso) or t[0])


def test_zona_liberada_apaga_la_alarma(camara, monkeypatch):
    hilo, eventos = camara
    _reloj(monkeypatch, 1.0)
    _simular(monkeypatch, personas=[EN_ZONA])
    for _ in range(3):
        hilo._procesar(NEGRO())
    _simular(monkeypatch)                               # se fue
    for _ in range(12):
        hilo._procesar(NEGRO())
    assert hilo.alarma == "alerta"                      # todavía no se sostuvo: puede ser un parpadeo
    for _ in range(30):
        hilo._procesar(NEGRO())
    assert [e[0] for e in eventos] == ["zona_prohibida", "con_casco"] and hilo.alarma is None


def test_parpadeo_con_el_caido_en_el_piso_no_apaga_la_sirena(camara, monkeypatch):
    hilo, eventos = camara
    _reloj(monkeypatch, 0.1)                            # 10 cuadros/s
    caido = [330, 300, 290, 100]                        # acostado, fuera de la zona
    _simular(monkeypatch, personas=[caido])
    for _ in range(110):
        hilo._procesar(NEGRO())
    assert [e[0] for e in eventos] == ["hombre_caido"]
    _simular(monkeypatch)                               # 0,5 s sin verlo (un compañero lo tapa)
    for _ in range(5):
        hilo._procesar(NEGRO())
    _simular(monkeypatch, personas=[caido])
    for _ in range(150):
        hilo._procesar(NEGRO())
    assert "con_casco" not in [e[0] for e in eventos] and hilo.alarma == "alerta"


def test_sin_alerta_nueva_no_se_codifica_la_foto(camara, monkeypatch):
    hilo, _ = camara
    _simular(monkeypatch)
    codificados = []
    original = motor.cv2.imencode
    monkeypatch.setattr(motor.cv2, "imencode", lambda *a: codificados.append(1) or original(*a))
    for _ in range(3):
        hilo._procesar(NEGRO())
    assert codificados == []


def test_sin_detector_de_cascos_el_panel_lo_dice(camara, monkeypatch):
    hilo, _ = camara
    _simular(monkeypatch)
    monkeypatch.setattr(motor.helmet_detector, "disponible", lambda: True)
    hilo._procesar(NEGRO())
    assert hilo.estado["titulo"] == "SIN PERSONAS"
    monkeypatch.setattr(motor.helmet_detector, "disponible", lambda: False)
    hilo._procesar(NEGRO())
    assert hilo.estado["titulo"] == "SIN DETECTOR DE CASCOS"


def test_caida_con_pose_no_pide_las_cajas_de_persona(monkeypatch):
    cam = data_store.create_camera("Taller", "", "usb", "0")     # sin zonas
    pedidos = []
    monkeypatch.setattr(motor, "analizar",
                        lambda img, modos, con_personas=False, **kw: pedidos.append(con_personas) or ([], []))
    monkeypatch.setattr(motor.salud, "lente", lambda img: None)
    hilo = motor.Camara(cam)
    hilo._procesar(NEGRO())                             # sin modelo de pose: hacen falta las cajas
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    monkeypatch.setattr(motor.pose, "detectar", lambda img: [])
    hilo._procesar(NEGRO())
    assert pedidos == [True, False]


def test_sin_senal_reintenta_si_no_se_pudo_registrar(monkeypatch):
    cam = data_store.create_camera("Rampa", "", "ip", "rtsp://10.0.0.9/stream")
    eventos, fallas = [], iter([PermissionError("archivo tomado")])

    def registrar(*a):
        for error in fallas:
            raise error
        eventos.append(a)

    monkeypatch.setattr(motor, "registrar_evento", registrar)
    hilo = motor.Camara(cam)
    for t in (0, 61, 62, 63):                           # en t = 61 falla al guardar
        hilo._sin_senal(t)
    assert eventos == [("camara_sin_senal", 100, cam["id"])]


def test_primer_pedido_pone_a_vigilar_las_camaras_relay(flask_app, client, monkeypatch):
    llamadas = []
    monkeypatch.setattr(motor, "sincronizar", lambda: llamadas.append(1))
    monkeypatch.setattr(flask_app, "_relays_vigilados", False)
    client.get("/admin/login")
    assert llamadas == []                               # en los tests no se enciende nada
    monkeypatch.setitem(flask_app.app.config, "TESTING", False)
    client.get("/admin/login")
    client.get("/admin/login")
    assert llamadas == [1]                              # una sola vez por proceso
