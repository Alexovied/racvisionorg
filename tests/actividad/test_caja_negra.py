"""CAJA NEGRA — pruebas funcionales contra los requerimientos de RAC Vision.

Se prueba solo por la interfaz pública (rutas HTTP del panel y de la API), sin
mirar cómo está hecho por dentro. Técnicas:
  - CRUD de cada entidad: Usuarios, Cámaras, Personas, Eventos, Configuración
  - Partición de equivalencia y valores límite:
        confianza del evento [0, 100]      · umbral de confianza [50, 99]
        frecuencia de escaneo [50, 1000]   · confirmaciones [1, 20]
        contraseña (≥ 4 caracteres)        · nombre de persona (2 a 60 letras)
        cantidad de fotos de una persona [3, 10]
  - Transición de estados: cámara activa ⇄ inactiva
  - Tabla de decisión: modos de /predict (casco / chaleco)
Los IDs (CN-xx) son los de la tabla de entrega/caja_negra.md.
"""
import base64
import io
from unittest import mock

import cv2
import numpy as np
import pytest

import data_store
import face_id

pytestmark = pytest.mark.caja_negra


def _texto(r):
    return r.get_data(as_text=True)


# ═══════════════════════════════════════════════════════════════════════════
#  USUARIOS (CRUD) — /admin/users
# ═══════════════════════════════════════════════════════════════════════════

def _alta_usuario(c, username="juan", password="clave1", role="operator"):
    return c.post("/admin/users", data={"action": "create", "username": username, "password": password,
                                         "role": role}, follow_redirects=True)


def test_CN01_alta_usuario_valido(admin_c):
    assert "creado exitosamente" in _texto(_alta_usuario(admin_c))
    assert data_store.get_user_by_username("juan")["role"] == "operator"


def test_CN02_alta_usuario_duplicado(admin_c):
    _alta_usuario(admin_c)
    assert "ya existe" in _texto(_alta_usuario(admin_c, password="otra-clave"))
    assert sum(u["username"] == "juan" for u in data_store.load_users()) == 1


@pytest.mark.parametrize("password, aceptada", [
    ("", False),          # partición inválida: vacía
    ("abc", False),       # límite inferior − 1 (3 caracteres)
    ("abcd", True),       # límite inferior (4)
    ("abcde", True),      # límite inferior + 1 (5)
])
def test_CN03_valores_limite_contrasena(admin_c, password, aceptada):
    _alta_usuario(admin_c, password=password)
    assert (data_store.get_user_by_username("juan") is not None) == aceptada


def test_CN04_alta_con_rol_inexistente(admin_c):
    assert "Rol inválido" in _texto(_alta_usuario(admin_c, role="superuser"))
    assert data_store.get_user_by_username("juan") is None


def test_CN05_consulta_lista_de_usuarios(admin_c):
    _alta_usuario(admin_c)
    html = _texto(admin_c.get("/admin/users"))
    assert "juan" in html and "admin_qa" in html


def test_CN06_modificacion_de_rol(admin_c):
    _alta_usuario(admin_c)
    uid = data_store.get_user_by_username("juan")["id"]
    assert "Usuario actualizado" in _texto(admin_c.post("/admin/users", data={
        "action": "edit", "user_id": uid, "role": "viewer"}, follow_redirects=True))
    assert data_store.get_user_by_username("juan")["role"] == "viewer"


def test_CN07_modificacion_a_nombre_ya_usado(admin_c):
    _alta_usuario(admin_c)
    uid = data_store.get_user_by_username("juan")["id"]
    r = admin_c.post("/admin/users", data={"action": "edit", "user_id": uid, "username": "admin_qa"},
                     follow_redirects=True)
    assert "ya está en uso" in _texto(r)
    assert data_store.get_user_by_username("juan") is not None


def test_CN08_baja_de_otro_usuario(admin_c):
    _alta_usuario(admin_c)
    uid = data_store.get_user_by_username("juan")["id"]
    admin_c.post(f"/admin/users/{uid}/delete")
    assert data_store.get_user_by_username("juan") is None


def test_CN09_baja_del_propio_usuario_rechazada(admin_c):
    uid = data_store.get_user_by_username("admin_qa")["id"]
    assert "No puedes eliminar tu propio usuario" in _texto(
        admin_c.post(f"/admin/users/{uid}/delete", follow_redirects=True))
    assert data_store.get_user_by_username("admin_qa") is not None


def test_CN10_operador_no_puede_crear_usuarios(operator_client):
    _alta_usuario(operator_client, username="intruso")
    assert data_store.get_user_by_username("intruso") is None


@pytest.mark.defecto("DEF-07")
def test_CN11_baja_de_usuario_inexistente_no_informa_exito(admin_c):
    # Esperado: avisar que el usuario no existe. Obtenido: "Usuario eliminado."
    r = admin_c.post("/admin/users/id-que-no-existe/delete", follow_redirects=True)
    assert "Usuario eliminado" not in _texto(r)


@pytest.mark.defecto("DEF-02")
def test_CN12_el_unico_admin_no_puede_quitarse_el_rol(admin_c):
    # Esperado: rechazar el cambio (si no, nadie puede volver a administrar el sistema).
    # Se borra el admin inicial que crea data_store para que admin_qa sea el único.
    for u in data_store.load_users():
        if u["username"] != "admin_qa":
            data_store.delete_user(u["id"])
    assert [u["username"] for u in data_store.load_users()] == ["admin_qa"]
    uid = data_store.get_user_by_username("admin_qa")["id"]
    admin_c.post("/admin/users", data={"action": "edit", "user_id": uid, "role": "operator"})
    assert any(u["role"] == "admin" for u in data_store.load_users())


# ═══════════════════════════════════════════════════════════════════════════
#  CÁMARAS (CRUD + transición de estados) — /admin/cameras
# ═══════════════════════════════════════════════════════════════════════════

def test_CN13_alta_camara_valida(admin_c):
    r = admin_c.post("/admin/cameras", data={"name": "Acceso Norte", "location": "Portón 1"}, follow_redirects=True)
    assert "registrada" in _texto(r)
    [cam] = data_store.load_cameras()
    assert (cam["name"], cam["location"], cam["active"]) == ("Acceso Norte", "Portón 1", True)


@pytest.mark.parametrize("nombre", ["", "    "])
def test_CN14_alta_camara_sin_nombre(admin_c, nombre):
    assert "obligatorio" in _texto(admin_c.post("/admin/cameras", data={"name": nombre}, follow_redirects=True))
    assert data_store.load_cameras() == []


def test_CN15_consulta_camaras(admin_c):
    data_store.create_camera("Taller", "Sector B")
    assert "Taller" in _texto(admin_c.get("/admin/cameras"))


def test_CN16_transicion_de_estados_activa_inactiva(admin_c):
    # Estados: ACTIVA --toggle--> INACTIVA --toggle--> ACTIVA
    cam = data_store.create_camera("Taller", "")
    assert "desactivada" in _texto(admin_c.post(f"/admin/cameras/{cam['id']}/toggle", follow_redirects=True))
    assert data_store.load_cameras()[0]["active"] is False
    assert "activada" in _texto(admin_c.post(f"/admin/cameras/{cam['id']}/toggle", follow_redirects=True))
    assert data_store.load_cameras()[0]["active"] is True


def test_CN17_baja_camara_por_admin(admin_c):
    cam = data_store.create_camera("Taller", "")
    admin_c.post(f"/admin/cameras/{cam['id']}/delete")
    assert data_store.load_cameras() == []


def test_CN18_operador_no_puede_borrar_camaras(operator_client):
    cam = data_store.create_camera("Taller", "")
    operator_client.post(f"/admin/cameras/{cam['id']}/delete")
    assert len(data_store.load_cameras()) == 1


@pytest.mark.defecto("DEF-07")
def test_CN19_baja_de_camara_inexistente_no_informa_exito(admin_c):
    r = admin_c.post("/admin/cameras/no-existe/delete", follow_redirects=True)
    assert "Cámara eliminada" not in _texto(r)


# ═══════════════════════════════════════════════════════════════════════════
#  EVENTOS — alta por API (partición de equivalencia + valores límite)
#  Campo: confidence, válido en [0, 100]
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("confianza, codigo", [
    (-0.1, 400),    # límite inferior − ε  (clase inválida: < 0)
    (0, 200),       # límite inferior
    (0.1, 200),     # límite inferior + ε
    (50, 200),      # valor típico de la clase válida
    (99.9, 200),    # límite superior − ε
    (100, 200),     # límite superior
    (100.1, 400),   # límite superior + ε (clase inválida: > 100)
    ("80", 400),    # clase inválida: texto
    (None, 400),    # clase inválida: falta el dato
])
def test_CN20_valores_limite_confianza_del_evento(operator_client, confianza, codigo):
    with mock.patch("admin.enviar_comando_esp32"):
        r = operator_client.post("/admin/api/events", json={"type": "con_casco", "confidence": confianza})
    assert r.status_code == codigo
    assert len(data_store.load_events()) == (1 if codigo == 200 else 0)


@pytest.mark.parametrize("tipo, codigo", [
    ("con_casco", 200), ("sin_casco", 200), ("sin_chaleco", 200),   # clases válidas
    ("CON_CASCO", 400), ("", 400), ("sin_lentes", 400),             # clases inválidas
])
def test_CN21_particion_tipo_de_evento(operator_client, tipo, codigo):
    with mock.patch("admin.enviar_comando_esp32"):
        assert operator_client.post("/admin/api/events", json={"type": tipo, "confidence": 80}).status_code == codigo


def test_CN22_consulta_auditoria_con_filtro(operator_client):
    data_store.append_event("con_casco", 90.5)
    data_store.append_event("sin_casco", 70.5)
    html = _texto(operator_client.get("/admin/audit?type=sin_casco"))
    assert "70.5%" in html and "90.5%" not in html


def test_CN23_exportar_auditoria_csv(operator_client):
    data_store.append_event("sin_casco", 77.5, persona="owen")
    r = operator_client.get("/admin/audit/export")
    assert r.mimetype == "text/csv" and "sin_casco,77.5" in _texto(r) and "owen" in _texto(r)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN (Read / Update / Reset) — valores límite de 3 campos numéricos
# ═══════════════════════════════════════════════════════════════════════════

def _guardar_config(c, **campos):
    form = {"action": "save", "min_confidence": 51, "confirm_threshold": 3, "scan_rate_ms": 100,
            "detectar_casco": "on", "detectar_chaleco": "on"}
    form.update(campos)
    return c.post("/admin/config", data=form, follow_redirects=True)


@pytest.mark.parametrize("campo, enviado, guardado", [
    ("min_confidence", 49, 50), ("min_confidence", 50, 50), ("min_confidence", 51, 51),
    ("min_confidence", 98, 98), ("min_confidence", 99, 99), ("min_confidence", 100, 99),
    ("confirm_threshold", 0, 1), ("confirm_threshold", 1, 1), ("confirm_threshold", 20, 20),
    ("confirm_threshold", 21, 20),
    ("scan_rate_ms", 49, 50), ("scan_rate_ms", 50, 50), ("scan_rate_ms", 1000, 1000),
    ("scan_rate_ms", 1001, 1000),
])
def test_CN24_valores_limite_configuracion(admin_c, campo, enviado, guardado):
    _guardar_config(admin_c, **{campo: enviado})
    assert data_store.load_config()[campo] == guardado


def test_CN25_configuracion_no_numerica_se_rechaza(admin_c):
    assert "Error en los valores" in _texto(_guardar_config(admin_c, min_confidence="alto"))
    assert data_store.load_config()["min_confidence"] == 51


def test_CN26_consulta_y_reset_de_configuracion(admin_c):
    _guardar_config(admin_c, min_confidence=80)
    assert admin_c.get("/admin/api/config").get_json()["min_confidence"] == 80
    admin_c.post("/admin/config", data={"action": "reset"})
    assert admin_c.get("/admin/api/config").get_json() == data_store.DEFAULT_CONFIG


# ═══════════════════════════════════════════════════════════════════════════
#  PERSONAS (reconocimiento facial) — valores límite de nombre y de fotos
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sface_falso(monkeypatch):
    monkeypatch.setattr(face_id, "disponible", lambda: True)
    monkeypatch.setattr(face_id, "embedding", lambda img: np.ones(128) / np.sqrt(128))


def _jpg():
    return io.BytesIO(cv2.imencode(".jpg", np.full((60, 60, 3), 200, np.uint8))[1].tobytes())


def _alta_persona(c, nombre="Ana", n_fotos=3):
    fotos = [(_jpg(), f"f{i}.jpg") for i in range(n_fotos)]
    return c.post("/admin/personas", data={"nombre": nombre, "consentimiento": "si", "fotos": fotos},
                  content_type="multipart/form-data", follow_redirects=True)


@pytest.mark.parametrize("nombre, aceptado", [
    ("A", False),            # 1 carácter (límite − 1)
    ("Al", True),            # 2 (límite inferior)
    ("A" * 60, True),        # 60 (límite superior)
    ("A" * 61, False),       # 61 (límite + 1)
    ("Ana 2", False),        # clase inválida: con números
    ("José O'Neil", True),   # clase válida: tildes, espacio y apóstrofo
])
def test_CN27_valores_limite_nombre_de_persona(admin_c, sface_falso, nombre, aceptado):
    _alta_persona(admin_c, nombre=nombre)
    assert (len(face_id.listar()) == 1) == aceptado


@pytest.mark.parametrize("n_fotos, aceptado", [(2, False), (3, True), (10, True), (11, False)])
def test_CN28_valores_limite_cantidad_de_fotos(admin_c, sface_falso, n_fotos, aceptado):
    _alta_persona(admin_c, n_fotos=n_fotos)
    assert (len(face_id.listar()) == 1) == aceptado


def test_CN29_baja_de_persona_inexistente(admin_c, sface_falso):
    r = admin_c.post("/admin/personas/borrar", data={"nombre": "Nadie"}, follow_redirects=True)
    assert "no está registrada" in _texto(r)


# ═══════════════════════════════════════════════════════════════════════════
#  TABLA DE DECISIÓN — ¿qué detector usa /predict?
#  Condiciones: detectar_casco · detectar_chaleco · modelo de chalecos disponible
# ═══════════════════════════════════════════════════════════════════════════

def _frame():
    ok, buf = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


CABEZA = {"bbox": [10, 20, 30, 40], "clase": "casco", "conf": 0.9, "p_casco": 0.9, "p_sin": 0.1}


@pytest.mark.parametrize("casco, chaleco, epp_ok, accion", [
    (True, True, True, "cabezas+chaleco"),     # R1
    (True, True, False, "cabezas+chaleco"),    # R2
    (True, False, True, "cabezas"),            # R3
    (True, False, False, "cabezas"),           # R4
    (False, True, True, "personas"),           # R5: solo chaleco con modelo propio
    (False, True, False, "cabezas_sin_casco"), # R6: solo chaleco, sin modelo → usa cabezas
])
def test_CN30_tabla_de_decision_predict(flask_app, operator_client, casco, chaleco, epp_ok, accion):
    data_store.save_config({**data_store.DEFAULT_CONFIG, "detectar_casco": casco, "detectar_chaleco": chaleco})
    persona = {"bbox": [5, 5, 100, 300], "chaleco": True}
    with mock.patch.object(flask_app.helmet_detector, "disponible", return_value=True), \
         mock.patch.object(flask_app.helmet_detector, "detectar", return_value=[CABEZA]), \
         mock.patch.object(flask_app.epp, "disponible", return_value=epp_ok), \
         mock.patch.object(flask_app.epp, "detectar_personas", return_value=[persona]), \
         mock.patch.object(flask_app.epp, "chalecos_por_cabeza", return_value=[True]):
        cara = operator_client.post("/predict", json={"image": _frame()}).get_json()["faces"][0]
    if accion == "personas":
        assert cara["bbox"] == [5, 5, 100, 300] and "con_casco" not in cara
    else:
        assert cara["bbox"] == [10, 20, 30, 40]
        assert ("con_casco" in cara) == (accion != "cabezas_sin_casco")
        assert ("chaleco" in cara) == (accion != "cabezas")
