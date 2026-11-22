"""CAJA BLANCA — pruebas estructurales sobre el código de RAC Vision.

Funciones elegidas (las de más lógica de decisión del backend):
  A. admin.validate_event_payload   → ruta básica (V(G) = 9) + condición múltiple
  B. epp.chalecos_por_cabeza        → ruta básica (V(G) = 5) + bucle
  C. admin._safe_next               → condición múltiple (4 condiciones con OR)
  D. data_store.get_filtered_events → condición múltiple (AND) + bucle simple
  E. data_store.append_event        → prueba de bucle simple (Beizer, n = 5000)
  F. data_store.update_user         → prueba de bucles anidados (Beizer)

Los IDs (CB-A-C1, ...) coinciden con las tablas de entrega/caja_blanca.md.
"""
import json
import math
from unittest import mock

import numpy as np
import pytest

import admin
import data_store
import epp

pytestmark = pytest.mark.caja_blanca


# ═══════════════════════════════════════════════════════════════════════════
#  A. validate_event_payload — un caso por cada camino independiente (V(G) = 9)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def camaras(tmp_data):
    """Dos cámaras registradas (el bucle del nodo 13 da 2 vueltas)."""
    a = data_store.create_camera("Acceso Norte", "Portón 1")
    b = data_store.create_camera("Taller", "Sector B")
    return a, b


def _err(resultado):
    return resultado[3]


def test_CB_A_C1_payload_que_no_es_objeto(tmp_data):
    # Camino 1: 1 → 2 → 18
    assert _err(admin.validate_event_payload(["no", "es", "dict"])) == "Se esperaba un objeto JSON"


def test_CB_A_C2_tipo_de_evento_invalido(tmp_data):
    # Camino 2: 1 → 3 → 4 → 18
    assert "'type' debe ser uno de" in _err(admin.validate_event_payload({"type": "casco_roto", "confidence": 80}))


def test_CB_A_C3_confianza_no_numerica(tmp_data):
    # Camino 3: 1 → 3 → 5 → 6 → 18
    assert _err(admin.validate_event_payload({"type": "sin_casco", "confidence": "80"})) == "'confidence' debe ser un número"


def test_CB_A_C4_confianza_fuera_de_rango(tmp_data):
    # Camino 4: 1 → 3 → 5 → 7 → 8 → 18
    assert _err(admin.validate_event_payload({"type": "sin_casco", "confidence": 150})) == "'confidence' debe estar entre 0 y 100"


def test_CB_A_C5_camara_que_no_es_texto(tmp_data):
    # Camino 5: 1 → 3 → 5 → 7 → 9 → 10 → 18
    assert _err(admin.validate_event_payload({"type": "sin_casco", "confidence": 80, "camera": "   "})) == "'camera' debe ser un texto"


def test_CB_A_C6_camara_por_defecto_webcam(tmp_data):
    # Camino 6: 1 → 3 → 5 → 7 → 9 → 11 → 17 → 18 (no entra al bucle)
    assert admin.validate_event_payload({"type": "con_casco", "confidence": 95}) == ("con_casco", 95.0, "webcam", None)


def test_CB_A_C7_camara_desconocida_sin_camaras_registradas(tmp_data):
    # Camino 7: … 11 → 12 → 13 → 15 → 16 → 18 (bucle salteado: 0 vueltas)
    assert _err(admin.validate_event_payload({"type": "sin_casco", "confidence": 80, "camera": "Galpón"})) \
        == "'camera' no corresponde a una cámara registrada"


def test_CB_A_C8_camara_desconocida_con_camaras_registradas(camaras):
    # Camino 8: … 12 → 13 → 14 → 13 → 15 → 16 → 18 (el bucle da vueltas y no la encuentra)
    assert _err(admin.validate_event_payload({"type": "sin_casco", "confidence": 80, "camera": "Galpón"})) \
        == "'camera' no corresponde a una cámara registrada"


def test_CB_A_C9_camara_registrada_por_nombre_o_id(camaras):
    # Camino 9: … 12 → 13 → 14 → 13 → 15 → 17 → 18
    a, _ = camaras
    assert admin.validate_event_payload({"type": "sin_casco", "confidence": 80, "camera": " Taller "})[2] == "Taller"
    assert admin.validate_event_payload({"type": "sin_casco", "confidence": 80, "camera": a["id"]})[2] == a["id"]


# ── Prueba de condición múltiple sobre el nodo 5 ────────────────────────────
#   isinstance(confidence, bool)  OR  not isinstance(confidence, (int, float))
#        (A)                               (B)
# En Python True es int, así que la combinación A=V, B=V es imposible.

@pytest.mark.parametrize("valor, a, b, rechaza", [
    (True, True, False, True),     # A verdadera sola → rechaza (un booleano no es una confianza)
    ("80", False, True, True),     # B verdadera sola → rechaza
    (80, False, False, False),     # ninguna → sigue
    (80.5, False, False, False),   # float, ninguna → sigue
])
def test_CB_A_condicion_multiple_tipo_de_confianza(tmp_data, valor, a, b, rechaza):
    assert isinstance(valor, bool) == a and (not isinstance(valor, (int, float))) == b
    err = _err(admin.validate_event_payload({"type": "sin_casco", "confidence": valor}))
    assert (err == "'confidence' debe ser un número") == rechaza


# ── Condición múltiple sobre el nodo 7:  isnan(c)  OR  not 0 <= c <= 100 ─────
# NaN hace verdaderas a las dos (cualquier comparación con NaN es falsa).

@pytest.mark.parametrize("valor, a, b, rechaza", [
    (float("nan"), True, True, True),
    (100.1, False, True, True),
    (-0.1, False, True, True),
    (0, False, False, False),
    (100, False, False, False),
])
def test_CB_A_condicion_multiple_rango_de_confianza(tmp_data, valor, a, b, rechaza):
    assert math.isnan(valor) == a and (not 0 <= valor <= 100) == b
    err = _err(admin.validate_event_payload({"type": "sin_casco", "confidence": valor}))
    assert (err == "'confidence' debe estar entre 0 y 100") == rechaza


# ═══════════════════════════════════════════════════════════════════════════
#  B. epp.chalecos_por_cabeza — V(G) = 5 (4 caminos factibles + 1 no factible)
# ═══════════════════════════════════════════════════════════════════════════

GRIS = (180, 180, 180)
AMARILLO_FLUOR = (40, 255, 200)


def _img():
    return np.full((480, 640, 3), GRIS, np.uint8)


@pytest.fixture
def modelo_falso(monkeypatch):
    """Simula el YOLO de chalecos: devuelve las detecciones que el test arme."""
    dets = []
    monkeypatch.setattr(epp, "_net", object())
    monkeypatch.setattr(epp.helmet_detector, "correr_yolo", lambda *a, **k: dets)
    return dets


PERSONA = np.array([0.9, 0.1])   # clase 0
CHALECO = np.array([0.1, 0.9])   # clase 1


def test_CB_B_C1_sin_modelo_usa_regla_de_color(monkeypatch):
    # Camino 1: 1 → 2 → 13. _net es None → regla de color para cada cabeza
    monkeypatch.setattr(epp, "_net", None)
    img = _img()
    img[110:250, 70:190] = AMARILLO_FLUOR
    assert epp.chalecos_por_cabeza(img, [[100, 40, 60, 60]]) == [True]


def test_CB_B_C1b_con_modelo_y_sin_cabezas(modelo_falso):
    # Mismo nodo 1, segunda condición del OR (not cabezas) → lista vacía.
    # El camino 1 → 3 → 4 → 12 (bucle con 0 vueltas) es NO FACTIBLE: si no hay
    # cabezas, el nodo 1 ya devolvió. Se documenta en caja_blanca.md.
    assert epp.chalecos_por_cabeza(_img(), []) == []


def test_CB_B_C2_cabeza_sin_persona_detectada_usa_color(modelo_falso):
    # Camino 2: 1 → 3 → 4 → 5 → 6 → 7 → 4 → 12 → 13
    assert epp.chalecos_por_cabeza(_img(), [[520, 40, 60, 60]]) == [False]


def test_CB_B_C3_persona_casi_sin_cuerpo_no_se_juzga(modelo_falso):
    # Camino 3: … 6 → 8 → 9 → 10 → 4 … (alto de la persona < 2 × alto de la cabeza)
    modelo_falso.append(([90, 30, 120, 100], PERSONA))
    assert epp.chalecos_por_cabeza(_img(), [[110, 40, 60, 60]]) == [None]


def test_CB_B_C4_persona_completa_con_chaleco_del_modelo(modelo_falso):
    # Camino 4: … 6 → 8 → 9 → 11 → 4 …
    modelo_falso.extend([([80, 30, 120, 300], PERSONA), ([95, 120, 90, 100], CHALECO)])
    assert epp.chalecos_por_cabeza(_img(), [[110, 40, 60, 60]]) == [True]


# ── Prueba de bucle simple (Beizer) sobre `for cabeza in cabezas` ──────────
# No hay un máximo n en el código: se prueba 0 (no factible con modelo, ver C1b),
# 1, 2 y m = 10 vueltas, y que cada cabeza conserve su posición en el resultado.

@pytest.mark.parametrize("vueltas", [1, 2, 10])
def test_CB_B_bucle_una_respuesta_por_cabeza(modelo_falso, vueltas):
    cabezas = [[10 + 60 * i, 40, 50, 50] for i in range(vueltas)]
    res = epp.chalecos_por_cabeza(_img(), cabezas)
    assert len(res) == vueltas and all(r is False for r in res)


# ═══════════════════════════════════════════════════════════════════════════
#  C. admin._safe_next — condición múltiple
#     not t  OR  not t.startswith("/")  OR  t.startswith("//")  OR  "\\" in t
#      (A)           (B)                       (C)                 (D)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destino, esperado, condicion", [
    ("", None, "A"),
    (None, None, "A"),
    ("https://evil.example.com", None, "B"),
    ("//evil.example.com", None, "C"),
    ("/\\evil.example.com", None, "D"),
    ("/admin/cameras", "/admin/cameras", "ninguna"),
    # Pasa las 4 condiciones, pero urlsplit borra el TAB y queda "//evil…":
    # lo frena el segundo if (scheme/netloc). Demuestra que ese if no es código muerto.
    ("/\t/evil.example.com", None, "segundo if"),
])
def test_CB_C_condicion_multiple_safe_next(destino, esperado, condicion):
    assert admin._safe_next(destino) == esperado


# ═══════════════════════════════════════════════════════════════════════════
#  D. get_filtered_events — condición múltiple con AND + bucle simple
#     event_type  AND  event_type != "all"  AND  e["type"] != event_type
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tres_eventos(tmp_data):
    data_store.append_event("con_casco", 90)
    data_store.append_event("sin_casco", 80)
    data_store.append_event("sin_chaleco", 70)


@pytest.mark.parametrize("filtro, cantidad", [
    (None, 3),          # A falsa → no filtra
    ("all", 3),         # A verdadera, B falsa → no filtra
    ("sin_casco", 1),   # A y B verdaderas; C verdadera para 2 eventos (se saltean) y falsa para 1
])
def test_CB_D_condicion_multiple_filtro_por_tipo(tres_eventos, filtro, cantidad):
    eventos, total = data_store.get_filtered_events(event_type=filtro)
    assert total == cantidad == len(eventos)


@pytest.mark.parametrize("n_eventos", [0, 1, 2, 24, 25, 26])
def test_CB_D_bucle_y_paginacion(tmp_data, n_eventos):
    # Bucle simple: 0, 1, 2 vueltas y alrededor del tamaño de página (24, 25, 26)
    for _ in range(n_eventos):
        data_store.append_event("con_casco", 90)
    pagina1, total = data_store.get_filtered_events(page=1, per_page=25)
    pagina2, _ = data_store.get_filtered_events(page=2, per_page=25)
    assert total == n_eventos
    assert len(pagina1) == min(n_eventos, 25)
    assert len(pagina2) == max(0, n_eventos - 25)


def test_CB_D_fecha_invalida_se_ignora(tres_eventos):
    # Rama except (ValueError): la fecha mal escrita no filtra ni rompe
    _, total = data_store.get_filtered_events(date_from="2026-99-99")
    assert total == 3


def test_CB_D_rango_de_fechas(tres_eventos):
    _, total = data_store.get_filtered_events(date_from="2000-01-01", date_to="2000-12-31")
    assert total == 0


# ═══════════════════════════════════════════════════════════════════════════
#  E. append_event — prueba de bucle simple (Beizer) con n = 5000 eventos
#     for viejo in events[5000:]: borrar su foto
#     Vueltas del bucle = eventos previos + 1 − 5000
# ═══════════════════════════════════════════════════════════════════════════

def _precargar(tmp_data, k):
    """Deja k eventos viejos en events.json, todos con foto en disco."""
    eventos = [{"id": f"viejo-{i:05d}", "type": "sin_casco", "confidence": 80.0, "camera": "webcam",
                "timestamp": "2026-01-01T00:00:00", "foto": True} for i in range(k)]
    (tmp_data / "alertas").mkdir(parents=True, exist_ok=True)
    for e in eventos[-3:]:   # solo los 3 más viejos tienen archivo (los que pueden salir)
        (tmp_data / "alertas" / f"{e['id']}.jpg").write_bytes(b"jpg")
    (tmp_data / "events.json").write_text(json.dumps(eventos))
    return eventos


@pytest.mark.parametrize("previos, vueltas", [
    (0, 0),        # saltear el bucle (lista vacía)
    (1, 0),
    (2, 0),
    (100, 0),      # m < n
    (4999, 0),     # n − 1: queda justo en 5000
    (5000, 1),     # n: sale 1 evento
    (5001, 2),     # n + 1: salen 2 eventos
])
def test_CB_E_bucle_de_retencion_5000(tmp_data, previos, vueltas):
    viejos = _precargar(tmp_data, previos)
    data_store.append_event("sin_casco", 90)
    eventos = data_store.load_events()
    assert len(eventos) == min(previos + 1, 5000)
    borrados = viejos[len(viejos) - vueltas:] if vueltas else []
    for e in borrados:
        assert not (tmp_data / "alertas" / f"{e['id']}.jpg").exists()
    # Los que siguen en la lista conservan su foto
    for e in viejos[-3:]:
        if e not in borrados:
            assert (tmp_data / "alertas" / f"{e['id']}.jpg").exists()


# ═══════════════════════════════════════════════════════════════════════════
#  F. update_user — bucles anidados (Beizer)
#     for u in users:                 (externo: busca el id)
#         for other in users:         (interno: ¿nombre repetido?)
#  Se empieza por el interno con el externo en su mínimo y se va hacia afuera.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("cantidad, posicion", [
    (1, 0),     # externo 1 vuelta, interno 1 vuelta (mínimos)
    (2, 0),     # interno 2 vueltas
    (10, 0),    # interno m vueltas, externo mínimo
    (10, 9),    # externo m vueltas (el buscado es el último)
])
def test_CB_F_bucles_anidados_update_user(tmp_data, cantidad, posicion):
    usuarios = [{"id": f"id-{i}", "username": f"user{i}", "password_hash": "x", "role": "operator"}
                for i in range(cantidad)]
    (tmp_data).mkdir(exist_ok=True)
    (tmp_data / "users.json").write_text(json.dumps(usuarios))
    assert data_store.update_user(f"id-{posicion}", username="nuevo") is True
    assert data_store.get_user_by_username("nuevo")["id"] == f"id-{posicion}"
    if cantidad > 1:   # el interno encuentra un duplicado → False y no cambia nada
        otro = (posicion + 1) % cantidad
        assert data_store.update_user(f"id-{otro}", username="nuevo") is False


def test_CB_F_id_inexistente_recorre_todo_y_devuelve_false(tmp_data):
    data_store.create_user("a", "clave")
    assert data_store.update_user("no-existe", username="b") is False
