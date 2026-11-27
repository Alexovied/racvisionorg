"""Arreglos de la auditoría en wearables, aprendizaje y riesgo: cada test falla sin su arreglo."""
from datetime import datetime

import pytest

import admin
import aprendizaje
import riesgo
import wearables


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    wearables._estados.clear()
    monkeypatch.setattr(admin, "_iniciar_vigilancia", lambda: None)   # sin hilos en los tests
    yield
    wearables._estados.clear()


def _mandar(client, token, **lectura):
    return client.post("/admin/api/dispositivos/telemetria", json=lectura,
                       headers={"Authorization": f"Bearer {token}"})


def _falla_una_vez(monkeypatch):
    """registrar_evento falla la primera vez (archivo tomado) y después anda."""
    real, fallas = admin.registrar_evento, iter([PermissionError("archivo tomado")])

    def registrar(*a, **kw):
        for error in fallas:
            raise error
        return real(*a, **kw)

    monkeypatch.setattr(admin, "registrar_evento", registrar)


# ── Si guardar falla, el aviso se reintenta ──────────────────────────────────

def test_sos_guardado_no_se_duplica_si_falla_un_aviso_posterior(client, monkeypatch):
    _, token = wearables.crear("Pulsera 01", "Juan")
    monkeypatch.setattr(admin.aviso_mqtt, "publicar", lambda e: (_ for _ in ()).throw(RuntimeError("can't start new thread")))
    assert _mandar(client, token, sos=True).json["alertas"] == ["boton_sos"]
    assert _mandar(client, token, sos=True).json["alertas"] == []   # ya avisado: no se guarda otro
    assert [e["type"] for e in admin.data_store.load_events()].count("boton_sos") == 1


def test_sos_que_no_se_pudo_guardar_se_reintenta(client, monkeypatch):
    _, token = wearables.crear("Pulsera 01", "Juan")
    _falla_una_vez(monkeypatch)
    with pytest.raises(PermissionError):
        _mandar(client, token, sos=True)
    assert _mandar(client, token, sos=True).json["alertas"] == ["boton_sos"]   # el botón sigue apretado


def test_sin_senal_que_no_se_pudo_guardar_se_reintenta_y_no_corta_a_los_demas(client, monkeypatch):
    tokens = [wearables.crear(f"Pulsera {i}", "Juan")[1] for i in (1, 2)]
    for token in tokens:
        _mandar(client, token, bpm=80)
    despues = max(e.ultima[0] for e in wearables._estados.values()) + wearables.SIN_SENAL_S + 1
    _falla_una_vez(monkeypatch)
    assert [e["camera"] for e in admin.revisar_wearables(despues)] == ["Pulsera 2"]
    assert [e["camera"] for e in admin.revisar_wearables(despues + 15)] == ["Pulsera 1"]
    assert admin.revisar_wearables(despues + 30) == []


# ── Reglas ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("inactivo_s", [30, 90, wearables.TOPES["inactivo_s"]])
def test_desmayo_avisa_con_cualquier_umbral_aprendido(inactivo_s):
    est, vistas = wearables.Estado(), []
    wearables.evaluar(est, {"caida": True, "actividad": 0.5}, 0, {"inactivo_s": inactivo_s})
    for t in range(5, 400, 5):
        vistas += wearables.evaluar(est, {"bpm": 80, "actividad": 0.01}, t, {"inactivo_s": inactivo_s})
    assert vistas == [("posible_desmayo", "quieto después de una caída")]


def test_pulso_que_oscila_en_el_umbral_avisa_una_sola_vez():
    est, vistas = wearables.Estado(), []
    for i in range(240):   # 20 minutos, una lectura cada 5 s
        vistas += wearables.evaluar(est, {"bpm": (161, 162, 163, 165, 164, 159)[i % 6]}, i * 5)
    assert [t for t, _ in vistas] == ["ritmo_cardiaco"]
    for t in range(1200, 1230, 5):   # 20 s de pulso normal: se rearma
        wearables.evaluar(est, {"bpm": 90}, t)
    for t in range(1230, 1260, 5):
        vistas += wearables.evaluar(est, {"bpm": 170}, t)
    assert len(vistas) == 2


@pytest.mark.parametrize("entre", [[(25, 120)], []])   # pasando por normal (no regresión) o directo
def test_despues_de_taquicardia_un_paro_avisa(entre):
    est, vistas = wearables.Estado(), []
    for t, bpm in [(0, 175), (10, 175), (20, 175), *entre, (30, 0), (40, 0), (50, 0)]:
        vistas += wearables.evaluar(est, {"bpm": bpm}, t)
    assert [d for _, d in vistas] == ["taquicardia 175 lpm", "sin pulso"]


def test_sensor_flojo_que_salta_entre_clases_avisa_una_sola_vez():
    est, vistas = wearables.Estado(), []
    for i in range(120):   # 10 minutos, una lectura cada 5 s
        vistas += wearables.evaluar(est, {"bpm": (70, 38, 0, 0, 0, 0)[i % 6]}, i * 5)
    assert vistas == [("ritmo_cardiaco", "sin pulso")]


def test_lecturas_cada_100_s_avisan_paro_y_desmayo():
    est, vistas = wearables.Estado(), []
    for i in range(10):   # cobertura marginal: llega una lectura cada 100 s
        vistas += wearables.evaluar(est, {"bpm": 0, "actividad": 0.01}, i * 100)
    assert sorted(t for t, _ in vistas) == ["posible_desmayo", "ritmo_cardiaco"]


def test_enlace_intermitente_no_repite_el_aviso_en_cada_reconexion():
    est, vistas, t = wearables.Estado(), [], 0
    for _ in range(6):   # 25 s con lecturas y 95 s de corte, taquicardia todo el tiempo
        for _ in range(6):
            vistas += wearables.evaluar(est, {"bpm": 170}, t)
            t += 5
        t += 95
    assert len(vistas) == 1


def test_una_lectura_al_volver_de_un_corte_no_cuenta_como_sostenida():
    est = wearables.Estado()
    assert wearables.evaluar(est, {"bpm": 175}, 0) == []
    assert wearables.evaluar(est, {"bpm": 175}, 7200) == []      # dos horas sin datos: empieza de nuevo
    assert wearables.evaluar(est, {"bpm": 175}, 7220) == [("ritmo_cardiaco", "taquicardia 175 lpm")]


def test_sensor_fuera_de_rango_no_tira_el_sos():
    assert wearables.validar_lectura({"sos": True, "spo2": 0, "bpm": 80}) == ({"sos": True, "bpm": 80.0}, None)
    assert "'spo2'" in wearables.validar_lectura({"spo2": 0})[1]   # sin nada más, sigue diciendo qué está mal


# ── Aprendizaje y riesgo ─────────────────────────────────────────────────────

def test_sugerencia_de_confianza_solo_mira_el_casco():
    marca = lambda tipo, conf, veredicto: {"type": tipo, "confidence": conf, "feedback": {"veredicto": veredicto}}
    eventos = [marca("sin_chaleco", 100, "real")] * 8 + [marca("sin_casco", 95, "real"), marca("sin_casco", 60, "falsa")]
    assert aprendizaje.sugerir_confianza(eventos, actual=51) is None   # solo 2 marcas de casco


def test_wearable_sin_senal_no_cuenta_como_alerta_de_seguridad():
    ahora = datetime(2026, 11, 22, 20, 0)
    eventos = [{"type": "wearable_sin_senal", "timestamp": "2026-11-22T18:00:00", "camera": "Pulsera 01"}]
    assert riesgo.kpis(eventos, ahora, dias=7)["dias_sin_alertas"] == 7
    assert riesgo.reporte(eventos, ahora.date())["total"] == 0
    assert riesgo.matriz(eventos, ahora, tipo="wearable_sin_senal")[6][18] == 1   # pedido por tipo, sí


def test_kpis_el_total_coincide_con_las_filas_por_dia():
    ahora = datetime(2026, 11, 22, 20, 0)
    eventos = [{"type": "sin_casco", "timestamp": "2026-11-15T21:00:00", "camera": "Portón"},
               {"type": "sin_casco", "timestamp": "2026-11-16T08:00:00", "camera": "Portón"}]
    k = riesgo.kpis(eventos, ahora, dias=7)
    assert k["total"] == sum(d["alertas"] for d in k["por_dia"]) == 1
    assert k["por_dia"][0]["fecha"] == "2026-11-16"


def test_pronostico_sin_eventos_dice_que_no_hay_datos():
    assert riesgo.pronostico([])["semanas_con_datos"] is False
