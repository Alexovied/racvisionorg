"""Revisión manual de alertas: foto limpia, corrección de etiquetas y dataset YOLO."""
import base64
import io
import json
import zipfile

import cv2
import numpy as np
import pytest

import data_store
import revision


def _jpg(valor=120, alto=480, ancho=640):
    ok, buf = cv2.imencode(".jpg", np.full((alto, ancho, 3), valor, np.uint8))
    return buf.tobytes()


def _data_url(valor=120):
    return "data:image/jpeg;base64," + base64.b64encode(_jpg(valor)).decode()


@pytest.fixture
def admin_client(flask_app):
    c = flask_app.app.test_client()
    r = c.post("/admin/login", data={"username": "admin", "password": "admin-solo-para-tests"})
    assert r.status_code == 302
    return c


@pytest.fixture
def sin_modelo(monkeypatch):
    """Cabezas que "propone el modelo": una con casco (la del falso positivo de pelo oscuro)."""
    import epp
    import helmet_detector
    monkeypatch.setattr(helmet_detector, "disponible", lambda: True)
    monkeypatch.setattr(helmet_detector, "detectar", lambda img: [
        {"bbox": [np.int64(320), np.int64(120), np.int64(64), np.int64(48)], "p_casco": np.float32(0.8),
         "p_sin": np.float32(0.2)}])
    monkeypatch.setattr(epp, "chalecos_por_cabeza", lambda img, cabezas, dets=None: [np.bool_(False)] * len(cabezas))


def test_alerta_del_navegador_guarda_la_foto_limpia(operator_client, tmp_data):
    cam = data_store.create_camera("Taller", "", "webcam")
    r = operator_client.post("/admin/api/events", json={
        "type": "sin_casco", "confidence": 91, "camera": cam["id"], "foto": _data_url(50),
        "foto_limpia": _data_url(200)})
    assert r.status_code == 200
    ev = r.get_json()["event"]
    m = revision.de_evento(ev["id"])
    assert (m["origen"], m["tipo"], m["camera"], m["cajas"]) == ("alerta", "sin_casco", "Taller", None)
    img = cv2.imread(str(tmp_data / "muestras" / f"{m['id']}.webp"))
    assert abs(int(img.mean()) - 200) < 3       # es la limpia, no la de los recuadros


def test_sin_foto_limpia_o_con_basura_no_guarda_nada(operator_client):
    for limpia in (None, "data:image/jpeg;base64,bm8gZXMgdW5hIGltYWdlbg==", "no"):
        r = operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80,
                                                            "camera": "webcam", "foto_limpia": limpia})
        assert r.status_code == 200
    assert revision.cargar() == []


def test_modo_anonimo_no_guarda_fotos_para_revisar(operator_client):
    data_store.save_config({**data_store.load_config(), "modo_anonimo": True})
    operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80, "camera": "webcam",
                                                    "foto_limpia": _data_url()})
    cam = data_store.create_camera("Taller", "", "webcam")
    r = operator_client.post("/admin/api/revision/capturar", json={"camera": cam["id"], "foto": _data_url()})
    assert r.status_code == 409
    assert revision.cargar() == []


def test_con_casco_no_se_guarda_para_revisar(tmp_data):
    import admin
    ev = data_store.append_event("con_casco", 100, "webcam")
    assert admin.guardar_muestra(ev, _jpg()) is None


def test_motor_guarda_el_cuadro_sin_recuadros(flask_app, monkeypatch):
    import motor
    cam = data_store.create_camera("Escritorio", "", "usb", "0")
    cam = data_store.set_zonas(cam["id"], [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]])
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([], [[100, 100, 80, 300]]))
    hilo = motor.Camara(cam)
    for _ in range(3):
        hilo._procesar(np.full((480, 640, 3), 90, np.uint8))
    [m] = revision.cargar()
    assert m["tipo"] == "zona_prohibida" and data_store.get_event(m["event_id"])["type"] == "zona_prohibida"
    img = cv2.imread(revision.ruta_foto(m["id"]))
    assert np.abs(img.astype(int) - 90).max() < 8      # sin el polígono ni el recuadro rojo


def test_revisar_corrige_etiquetas_y_marca_la_alerta(operator_client, sin_modelo):
    ev = data_store.append_event("sin_casco", 70, "webcam")
    m = revision.guardar(_jpg(), "Taller", "alerta", "sin_casco", ev["id"])

    r = operator_client.get(f"/admin/api/revision/{m['id']}")
    dets = r.get_json()["detecciones"]
    assert dets == [{"bbox": [0.5, 0.25, 0.1, 0.1], "casco": "con", "confianza": 80.0, "chaleco": "sin"}]

    # Era pelo oscuro: la cabeza no tenía casco; y se suma una que el modelo no vio
    cajas = [{"bbox": [0.5, 0.25, 0.1, 0.1], "casco": "sin", "chaleco": "sin"},
             {"bbox": [0.1, 0.2, 0.08, 0.1], "casco": "con", "chaleco": None}]
    r = operator_client.post(f"/admin/api/revision/{m['id']}", json={"cajas": cajas, "veredicto": "real"})
    assert r.status_code == 200 and r.get_json()["resumen"]["revisadas"] == 1
    guardada = revision.obtener(m["id"])
    assert guardada["cajas"] == cajas and guardada["revisado_por"] == "operador_test"
    assert data_store.get_event(ev["id"])["feedback"]["veredicto"] == "real"   # también en Auditoría

    pendientes = operator_client.get("/admin/api/revision").get_json()["muestras"]
    revisadas = operator_client.get("/admin/api/revision?estado=revisadas").get_json()["muestras"]
    assert pendientes == [] and [x["id"] for x in revisadas] == [m["id"]]


@pytest.mark.parametrize("cajas", [
    None, "x", [{"bbox": [0.1, 0.1, 0.1]}], [{"bbox": [1.2, 0.1, 0.1, 0.1]}],
    [{"bbox": [0.1, 0.1, 0.1, 0.1], "casco": "quizas"}], [{"bbox": [0.1, 0.1, 0, 0.1]}],
    [{"bbox": [True, 0.1, 0.1, 0.1]}], [{"bbox": [0.1, 0.1, 0.1, 0.1]}] * 51,
])
def test_cajas_invalidas(operator_client, cajas):
    m = revision.guardar(_jpg(), "Taller", "captura")
    r = operator_client.post(f"/admin/api/revision/{m['id']}", json={"cajas": cajas})
    assert r.status_code == 400
    assert revision.obtener(m["id"])["cajas"] is None


def test_alerta_pide_veredicto_y_captura_no(operator_client):
    alerta = revision.guardar(_jpg(), "Taller", "alerta", "sin_casco", data_store.append_event("sin_casco", 70, "w")["id"])
    assert operator_client.post(f"/admin/api/revision/{alerta['id']}", json={"cajas": []}).status_code == 400
    captura = revision.guardar(_jpg(), "Taller", "captura")
    r = operator_client.post(f"/admin/api/revision/{captura['id']}", json={"cajas": [], "veredicto": "real"})
    assert r.status_code == 200 and revision.obtener(captura["id"])["veredicto"] is None


def test_caja_que_se_sale_se_recorta():
    [c] = revision.validar_cajas([{"bbox": [0.9, 0.9, 0.5, 0.5], "casco": "sin"}])
    assert c["bbox"] == [0.9, 0.9, 0.1, 0.1] and c["chaleco"] is None


def test_capturar_cuadro_de_webcam_y_de_camara_del_servidor(operator_client, monkeypatch):
    import motor
    web = data_store.create_camera("Webcam", "", "webcam")
    ip = data_store.create_camera("Portón", "", "ip", "rtsp://x")
    assert operator_client.post("/admin/api/revision/capturar", json={"camera": web["id"], "foto": _data_url()}).status_code == 200
    monkeypatch.setattr(motor, "cuadro_limpio", lambda cam_id: _jpg() if cam_id == ip["id"] else None)
    assert operator_client.post("/admin/api/revision/capturar", json={"camera": ip["id"]}).status_code == 200
    assert operator_client.post("/admin/api/revision/capturar", json={"camera": "nada"}).status_code == 404
    assert [(m["origen"], m["camera"]) for m in revision.cargar()] == [("captura", "Portón"), ("captura", "Webcam")]


def test_viewer_no_puede_revisar(flask_app):
    data_store.create_user("mira", "clave-mira", role="viewer")
    c = flask_app.app.test_client()
    c.post("/admin/login", data={"username": "mira", "password": "clave-mira"})
    m = revision.guardar(_jpg(), "Taller", "captura")
    assert c.get("/admin/api/revision").status_code == 403
    assert c.post(f"/admin/api/revision/{m['id']}", json={"cajas": []}).status_code == 403
    assert c.get(f"/admin/revision/{m['id']}.webp").status_code in (302, 403)


def test_foto_y_borrar(operator_client, tmp_data):
    m = revision.guardar(_jpg(), "Taller", "captura")
    r = operator_client.get(f"/admin/revision/{m['id']}.webp")
    assert r.status_code == 200 and r.data[:4] == b"RIFF"
    assert operator_client.get("/admin/revision/../../users.webp").status_code == 404
    assert operator_client.post(f"/admin/api/revision/{m['id']}/borrar").status_code == 200
    assert revision.cargar() == [] and not (tmp_data / "muestras" / f"{m['id']}.webp").exists()
    assert operator_client.get(f"/admin/revision/{m['id']}.webp").status_code == 404


def test_al_llenarse_se_borran_primero_las_viejas_sin_revisar(monkeypatch):
    monkeypatch.setattr(revision, "MAXIMO", 3)
    revisada = revision.guardar(_jpg(), "c", "captura")
    revision.revisar(revisada["id"], [], None, "yo")
    vieja = revision.guardar(_jpg(), "c", "captura")
    for _ in range(2):
        revision.guardar(_jpg(), "c", "captura")
    ids = [m["id"] for m in revision.cargar()]
    assert len(ids) == 3 and revisada["id"] in ids and vieja["id"] not in ids


def test_dataset_yolo(admin_client, operator_client):
    sin = revision.guardar(_jpg(), "Taller", "captura")
    revision.revisar(sin["id"], [{"bbox": [0.5, 0.25, 0.1, 0.2], "casco": "sin", "chaleco": None}], None, "yo")
    dos = revision.guardar(_jpg(), "Taller", "alerta", "sin_casco", data_store.append_event("sin_casco", 70, "w")["id"])
    revision.revisar(dos["id"], [{"bbox": [0.1, 0.1, 0.1, 0.1], "casco": "con", "chaleco": "con"},
                                 {"bbox": [0.6, 0.1, 0.1, 0.1], "casco": None, "chaleco": "sin"}], "falsa", "yo")
    vacia = revision.guardar(_jpg(), "Taller", "captura")      # nadie: ejemplo negativo
    revision.revisar(vacia["id"], [], None, "yo")
    revision.guardar(_jpg(), "Taller", "captura")              # sin revisar: no va

    assert operator_client.get("/admin/revision/dataset.zip").status_code == 302   # solo admin
    r = admin_client.get("/admin/revision/dataset.zip")
    assert r.status_code == 200 and r.mimetype == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.data))
    nombres = set(z.namelist())
    n = lambda m: f"rac_{m['id'][:8]}"
    assert {f"images/{n(m)}.jpg" for m in (sin, dos, vacia)} <= nombres
    assert z.read(f"labels/{n(sin)}.txt").decode() == "1 0.550000 0.350000 0.100000 0.200000"
    assert z.read(f"labels/{n(dos)}.txt").decode() == "0 0.150000 0.150000 0.100000 0.100000"   # "no se ve" afuera
    assert z.read(f"labels/{n(vacia)}.txt").decode() == ""
    assert f"sin_casco/{n(sin)}.jpg" in nombres and not any(x.startswith("con_casco/") for x in nombres)
    assert "names:\n  0: casco\n  1: sin_casco" in z.read("data.yaml").decode()
    filas = z.read("etiquetas.csv").decode().splitlines()
    assert len(filas) == 1 + 1 + 2 + 1 and "falsa" in filas[2]
    assert cv2.imdecode(np.frombuffer(z.read(f"images/{n(sin)}.jpg"), np.uint8), cv2.IMREAD_COLOR).shape == (480, 640, 3)


def test_dataset_vacio_vuelve_a_la_pagina(admin_client):
    r = admin_client.get("/admin/revision/dataset.zip")
    assert r.status_code == 302 and r.location.endswith("/admin/revision")


def test_pagina_de_revision(operator_client, sin_modelo):
    revision.guardar(_jpg(), "Taller", "captura")
    r = operator_client.get("/admin/revision")
    assert r.status_code == 200 and "Revisar alertas" in r.get_data(as_text=True)


def test_deshacer_revision(operator_client):
    ev = data_store.append_event("sin_casco", 70, "webcam")
    m = revision.guardar(_jpg(), "Taller", "alerta", "sin_casco", ev["id"])
    operator_client.post(f"/admin/api/revision/{m['id']}", json={"cajas": [], "veredicto": "falsa"})
    assert data_store.get_event(ev["id"])["feedback"]["veredicto"] == "falsa"
    r = operator_client.post(f"/admin/api/revision/{m['id']}/reabrir")
    assert r.status_code == 200 and r.get_json()["resumen"]["pendientes"] == 1
    m = revision.obtener(m["id"])
    assert (m["cajas"], m["veredicto"], m["revisado_por"]) == (None, None, None)
    assert "feedback" not in data_store.get_event(ev["id"])
    assert operator_client.post(f"/admin/api/revision/{'0' * 8}-0000-0000-0000-{'0' * 12}/reabrir").status_code == 404
