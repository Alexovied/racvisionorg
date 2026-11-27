"""Motor de cámaras del servidor: reglas de alerta, zonas de exclusión y hombre caído."""
import re
import threading

import cv2
import numpy as np
import pytest

import data_store
import motor


def test_un_solo_frame_malo_no_dispara_la_alerta():
    v = motor.Ventana()
    assert v.votar("falta", 3) is None and v.votar("falta", 3) is None
    assert v.votar("falta", 3) == "falta"      # recién al tercero
    assert v.votar("falta", 3) is None         # ya estaba en alerta: no se repite


def test_ok_necesita_la_ventana_llena():
    v = motor.Ventana()
    assert [v.votar("ok", 3) for _ in range(motor.VENTANA)] == [None] * (motor.VENTANA - 1) + ["ok"]


def test_zona_alerta_y_se_rearma_cuando_queda_libre():
    s = motor.Seguimiento()
    paso = lambda zona: s.procesar({"zona": zona}, False, 3, 10, 0.0)
    assert [paso("falta") for _ in range(3)] == [[], [], ["zona"]]
    assert s.resumen(["zona"]) == ("alerta", "ZONA PROHIBIDA")
    for _ in range(motor.VENTANA):
        paso("ok")
    assert s.resumen(["zona"]) == (None, None)
    # Otra intrusión vuelve a avisar: con la ventana llena de "ok" hacen falta 5 de 8 votos (60 %)
    assert [paso("falta") for _ in range(5)] == [[], [], [], [], ["zona"]]


def test_hombre_caido_avisa_una_vez_pasado_el_tiempo():
    s = motor.Seguimiento()
    paso = lambda caido, t: s.procesar({}, caido, 3, 10, t)
    assert [paso(True, 100 + t) for t in range(5)] == [[]] * 5
    assert paso(False, 105) == []                 # un frame sin verlo no corta la cuenta
    assert [paso(True, 106 + t) for t in range(5)] == [[]] * 4 + [["caida"]]   # t = 110
    assert paso(True, 111) == []                  # sigue caído: no se repite
    assert s.resumen([])[0] == "alerta"
    assert [paso(False, 112 + t) for t in range(4)] == [[]] * 4 and s.caido is False   # se levantó
    assert paso(True, 116) == [] and s.caido_desde is None    # un frame suelto no arranca la cuenta


def test_hombre_caido_no_cuenta_frames_sueltos_ni_huecos_de_video():
    s = motor.Seguimiento()
    # Acostado en 1 de cada 10 frames durante 30 s: no es una caída
    assert not any(s.procesar({}, i % 10 == 0, 3, 10, 100 + i * 0.1) for i in range(300))
    # Cámara trabada 15 s entre dos frames: ese tiempo no cuenta como tiempo en el suelo
    s = motor.Seguimiento()
    assert s.procesar({}, True, 3, 10, 100) == [] and s.procesar({}, True, 3, 10, 115) == []


def test_acostada_solo_cajas_anchas_enteras_y_sin_cabeza_grande():
    acostada = lambda p, cabezas=(): motor._acostada(p, cabezas, 640, 480)
    assert acostada([100, 300, 300, 100])
    assert acostada([100, 300, 300, 100], [[110, 310, 60, 60]])          # cabeza chica en una punta
    assert not acostada([100, 100, 80, 200])                             # de pie
    assert not acostada([100, 380, 300, 100])                            # cortada por el borde de abajo
    assert not acostada([0, 95, 605, 317])                               # cortada por el costado
    assert not acostada([20, 95, 600, 317], [[55, 48, 300, 328]])        # sentado cerca: cabeza enorme


def test_hombre_caido_registra_evento(camara, monkeypatch):
    hilo, eventos = camara
    reloj = iter(range(100, 200))
    monkeypatch.setattr(motor.time, "monotonic", lambda: next(reloj))
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([], [[330, 300, 290, 100]]))
    for _ in range(11):                # t = 100 ... 110 con caida_segundos = 10
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert [e[0] for e in eventos] == ["hombre_caido"]
    assert hilo.estado["titulo"] == "HOMBRE CAÍDO"


def test_zona_hasta_el_borde_incluye_a_quien_queda_cortado_abajo(monkeypatch):
    cam = data_store.create_camera("Escritorio", "", "usb", "0")
    cam = data_store.set_zonas(cam["id"], [[[0.05, 0.4], [0.95, 0.4], [0.95, 0.995], [0.05, 0.995]]])
    eventos = []
    monkeypatch.setattr(motor, "registrar_evento", lambda *a: eventos.append(a))
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([], [[200, 100, 200, 380]]))
    hilo = motor.Camara(cam)
    for _ in range(3):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))   # la caja termina en y = 480, el borde
    assert [e[0] for e in eventos] == ["zona_prohibida"]


def test_evento_que_no_se_pudo_guardar_se_reintenta(camara, monkeypatch):
    hilo, eventos = camara
    fallas = iter([PermissionError("archivo tomado")])

    def registrar(*a):
        for error in fallas:
            raise error
        eventos.append(a)

    monkeypatch.setattr(motor, "registrar_evento", registrar)
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([], [[100, 100, 80, 300]]))
    frame = lambda: hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    frame(), frame()
    with pytest.raises(PermissionError):
        frame()                        # tercer voto: decide la alerta y falla al guardar
    frame()
    assert [e[0] for e in eventos] == ["zona_prohibida"]


def test_evento_de_zona_lleva_el_nombre_de_quien_entro(camara, monkeypatch):
    hilo, eventos = camara
    cara = {"bbox": [110, 110, 50, 50], "nombre": "José", "con_casco": 95.0, "sin_casco": 5.0, "chaleco": True}
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([cara], [[100, 100, 80, 300]]))
    for _ in range(3):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert eventos[0][:4] == ("zona_prohibida", 100.0, hilo.cam["id"], "José")


def test_solo_el_operador_registra_y_activa_camaras(client):
    data_store.create_user("mirón", "clave-viewer", role="viewer")
    client.post("/admin/login", data={"username": "mirón", "password": "clave-viewer"})
    assert client.post("/admin/cameras", data={"name": "X", "type": "ip", "url": "rtsp://x"}).status_code == 403
    cam = data_store.create_camera("Entrada", "", "usb", "0")
    client.post(f"/admin/cameras/{cam['id']}/toggle")
    assert data_store.load_cameras() == [cam]
    assert client.get("/admin/cameras").status_code == 200      # el listado lo sigue viendo


def test_camara_ip_exige_una_direccion_de_video(operator_client):
    operator_client.post("/admin/cameras", data={"name": "Mala", "type": "ip", "url": "C:/Windows/win.ini"})
    operator_client.post("/admin/cameras", data={"name": "Buena", "type": "ip", "url": "rtsp://10.0.0.5/stream"})
    assert [c["name"] for c in data_store.load_cameras()] == ["Buena"]


@pytest.fixture
def camara(monkeypatch):
    """Cámara con una zona en la mitad izquierda, sin video ni modelos: analizar() está simulado."""
    cam = data_store.create_camera("Portón", "Sector A", "usb", "0")
    cam = data_store.set_zonas(cam["id"], [[[0, 0], [0.5, 0], [0.5, 1], [0, 1]]])
    eventos = []
    monkeypatch.setattr(motor, "registrar_evento", lambda *a: eventos.append(a))
    return motor.Camara(cam), eventos


def test_persona_en_zona_registra_evento_con_foto(camara, monkeypatch):
    hilo, eventos = camara
    en_zona = [100, 100, 80, 300]     # apoya en x=140: adentro
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([], [en_zona]))
    for _ in range(3):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert [e[0] for e in eventos] == ["zona_prohibida"]
    tipo, conf, cam_id, persona, foto, pos = eventos[0]
    assert cam_id == hilo.cam["id"] and foto[:2] == b"\xff\xd8"     # JPEG del frame con recuadros
    assert pos == [140 / 640, 400 / 480]                            # los pies de la persona, para el mapa de riesgo
    assert hilo.estado == {"clase": "alerta", "titulo": "ZONA PROHIBIDA"}


def test_persona_fuera_de_zona_no_alerta(camara, monkeypatch):
    hilo, eventos = camara
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([], [[400, 100, 80, 300]]))
    for _ in range(10):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert eventos == []


def test_sin_casco_pasa_de_alerta_a_todos_con_epp(camara, monkeypatch):
    hilo, eventos = camara
    cara = {"bbox": [300, 50, 60, 60], "nombre": "José", "con_casco": 5.0, "sin_casco": 95.0, "chaleco": True}
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([cara], []))
    for _ in range(3):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert eventos[0][:4] == ("sin_casco", 95.0, hilo.cam["id"], "José")
    cara.update(con_casco=95.0, sin_casco=5.0)
    for _ in range(motor.VENTANA):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert eventos[-1][0] == "con_casco"


def test_registrar_evento_de_zona_guarda_persona_y_foto(flask_app):
    cam = data_store.create_camera("Portón", "Sector A", "usb", "0")
    import admin
    jpg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()  # como el cuadro del motor
    ev = admin.registrar_evento("zona_prohibida", 100, cam["id"], None, jpg)
    assert (ev["type"], ev["camera"], ev["ubicacion"], ev["persona"], ev["foto"]) == \
           ("zona_prohibida", "Portón", "Sector A", "Sin identificar", True)
    assert data_store.get_stats()["alerts_24h"] == 1


def test_api_zonas_valida_y_guarda(operator_client):
    cam = data_store.create_camera("Portón", "", "usb", "0")
    url = f"/admin/api/cameras/{cam['id']}/zonas"
    zona = [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]
    assert operator_client.post(url, json={"zonas": [zona]}).status_code == 200
    assert data_store.load_cameras()[0]["zonas"] == [zona]
    for malas in ([[[0, 0], [1, 1]]], [[[0, 0], [2, 0], [1, 1]]], "x", [[[0, 0], [1, True], [1, 1]]]):
        assert operator_client.post(url, json={"zonas": malas}).status_code == 400
    assert operator_client.post("/admin/api/cameras/no-existe/zonas", json={"zonas": []}).status_code == 404


def test_frame_y_estado_sin_motor(operator_client):
    assert operator_client.get("/admin/api/cameras/x/frame.jpg").status_code == 404
    estado = operator_client.get("/admin/api/motor").get_json()
    assert set(estado) == {"_nodo"} and set(estado["_nodo"]) == {"cpu", "memoria", "temperatura"}


def test_camara_usb_exige_numero_de_dispositivo(operator_client):
    operator_client.post("/admin/cameras", data={"name": "Mala", "type": "usb", "url": "rtsp://x"})
    operator_client.post("/admin/cameras", data={"name": "Buena", "type": "usb"})
    assert [(c["name"], c["url"]) for c in data_store.load_cameras()] == [("Buena", "0")]


def test_eventos_simultaneos_no_se_pierden():
    hilos = [threading.Thread(target=data_store.append_event, args=("sin_casco", 90, f"cam{i}")) for i in range(20)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert len(data_store.load_events()) == 20


# ── Pose ─────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def sin_pose(monkeypatch):
    """Por defecto los tests usan la regla de la caja, esté o no detector_pose.onnx en la carpeta."""
    monkeypatch.setattr(motor.pose, "_net", None)


def _puntos(hombros, caderas, conf=0.9):
    p = np.zeros((17, 3), np.float32)
    for i in (5, 6):
        p[i] = (*hombros, conf)
    for i in (11, 12):
        p[i] = (*caderas, conf)
    return p


def test_torso_caido_por_angulo():
    caido = motor.pose.torso_caido
    assert caido(_puntos((300, 100), (300, 250))) is False        # de pie
    assert caido(_puntos((300, 100), (340, 250))) is False        # inclinado 15°
    assert caido(_puntos((100, 300), (300, 320))) is True         # acostado
    assert caido(_puntos((300, 300), (300, 150))) is True         # cabeza abajo
    assert caido(_puntos((300, 100), (300, 250), conf=0.2)) is None   # no se ven hombros / caderas


def test_torso_caido_con_caderas_tapadas_usa_la_cabeza():
    caido = motor.pose.torso_caido
    tapado = _puntos((300, 100), (300, 250))
    for i in (11, 12):
        tapado[i] = (300, 250, 0.1)        # frazada o borde de la imagen: caderas sin ver
    tapado[0] = (300, 40, 0.9)             # nariz arriba de los hombros: de pie
    assert caido(tapado) is False
    tapado[0] = (180, 105, 0.9)            # nariz al costado de los hombros: acostado
    assert caido(tapado) is True
    tapado[0] = (180, 105, 0.1)            # sin cabeza ni caderas no se decide
    assert caido(tapado) is None


def test_con_pose_sentado_cerca_no_es_caida_y_acostado_si(camara, monkeypatch):
    hilo, eventos = camara
    reloj = iter(range(100, 200))
    monkeypatch.setattr(motor.time, "monotonic", lambda: next(reloj))
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    ancha = [330, 95, 290, 200]      # caja ancha: la regla de la caja diría "acostado"
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([], [ancha]))
    monkeypatch.setattr(motor.pose, "detectar", lambda img: [(ancha, _puntos((470, 120), (470, 280)))])
    for _ in range(12):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert eventos == []                                          # torso derecho: no hay caída
    monkeypatch.setattr(motor.pose, "detectar", lambda img: [(ancha, _puntos((350, 200), (560, 210)))])
    for _ in range(16):    # 5 frames para ser mayoría en la ventana + 10 s acostado
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert [e[0] for e in eventos] == ["hombre_caido"]


def test_caida_con_pose_aunque_no_haya_modelo_de_chalecos(monkeypatch):
    # Sin detector_chaleco.onnx `personas` llega vacío: la pose igual tiene que ver al acostado
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    caja = [330, 195, 290, 100]
    cuerpos = [(caja, _puntos((350, 200), (560, 210)))]
    assert motor.acostados(np.zeros((480, 640, 3), np.uint8), [], [], {"casco": True, "chaleco": False},
                           cuerpos) == [caja]


def test_hilo_de_camara_sobrevive_a_un_error_de_opencv(camara, monkeypatch):
    hilo, _ = camara
    llamadas = []

    def abrir():
        llamadas.append(1)
        if len(llamadas) == 2:
            hilo.parar.set()
        raise cv2.error("driver roto")

    monkeypatch.setattr(hilo, "_abrir", abrir)
    monkeypatch.setattr(hilo.parar, "wait", lambda s=None: None)
    monkeypatch.setattr(motor.threading, "Thread", lambda **kw: mock_hilo())
    hilo.run()
    assert len(llamadas) == 2 and hilo.error == "Sin señal"     # reintentó en vez de morir


class mock_hilo:
    def start(self):
        pass


def test_camara_relay_recibe_cuadros_solo_con_su_token(operator_client, client, monkeypatch):
    recibidos = []
    monkeypatch.setattr(motor, "recibir", lambda cid, img, n=None: recibidos.append((cid, img.shape)) or True)
    r = operator_client.post("/admin/cameras", data={"name": "Oryx", "type": "relay"}, follow_redirects=True)
    cam = data_store.load_cameras()[0]
    token = re.search(r"--token (rac_\S+)", r.get_data(as_text=True)).group(1)
    assert cam["type"] == "relay" and token not in str(cam)      # solo se guarda el hash
    _, jpg = cv2.imencode(".jpg", np.zeros((1080, 1920, 3), np.uint8))
    url = f"/admin/api/cameras/{cam['id']}/cuadro"
    assert client.post(url, data=jpg.tobytes(), headers={"Authorization": "Bearer rac_otro"}).status_code == 401
    assert client.post(url, data=b"no es jpeg", headers={"Authorization": f"Bearer {token}"}).status_code == 400
    assert client.post(url, data=jpg.tobytes(), headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert recibidos == [(cam["id"], (1080, 1920, 3))]


def test_camara_relay_achica_y_queda_sin_senal_si_no_llegan_cuadros(monkeypatch):
    cam = data_store.create_camera("Oryx", "", "relay", "", token_hash="x")
    hilo = motor.Camara(cam)
    hilo.recibir(np.zeros((1080, 1920, 3), np.uint8))
    assert hilo.crudo.shape == (540, 960, 3)
    hilo.t_recibido -= motor.TIMEOUT_IP_MS / 1000 + 1
    espera = iter([False, True])
    monkeypatch.setattr(hilo.parar, "wait", lambda s: next(espera))
    hilo._esperar_relay()
    assert hilo.crudo is None and hilo.error == "Sin señal"


def test_camara_relay_descarta_cuadros_que_llegan_desordenados():
    hilo = motor.Camara(data_store.create_camera("Oryx", "", "relay", "", token_hash="x"))
    nuevo, viejo = np.zeros((10, 10, 3), np.uint8), np.ones((10, 10, 3), np.uint8)
    hilo.recibir(nuevo, 7)
    hilo.recibir(viejo, 6)
    assert hilo.crudo is nuevo
    hilo.recibir(viejo, 1000)   # relay reiniciado o salto: se toma
    hilo.recibir(nuevo, 1)      # el relay volvió a contar desde 1
    assert hilo.crudo is nuevo


def test_tirado_en_diagonal_no_se_endereza_como_camara_girada(monkeypatch):
    # Torso a ~55° de la vertical: con un giro "aprendido" de -28° quedaría en ~-27° (de pie)
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    caja = [200, 150, 260, 200]
    cuerpos = [(caja, _puntos((250, 180), (400, 285)))]
    img = np.zeros((480, 640, 3), np.uint8)
    assert motor.pose.torso_caido(cuerpos[0][1], -28.0) is False
    assert motor.acostados(img, [], [], {"casco": True, "chaleco": True}, cuerpos, -28.0) == [caja]
