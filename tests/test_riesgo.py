"""Mapa de riesgo, pronóstico, reporte diario y alertas por Telegram."""
import json
import os
from datetime import datetime, timedelta
from unittest import mock

import cv2
import numpy as np
import pytest

import admin
import aviso_telegram
import data_store
import motor
import riesgo

AHORA = datetime(2026, 11, 22, 20, 0)   # domingo


def ev(tipo, cuando, camara="Portón", **extra):
    return {"type": tipo, "timestamp": cuando.isoformat(), "camera": camara, "confidence": 90, **extra}


# ── riesgo.py ────────────────────────────────────────────────────────────────

def test_matriz_cuenta_solo_alertas_por_dia_y_hora():
    eventos = [ev("sin_casco", datetime(2026, 11, 20, 9, 15)),      # viernes 9 h
               ev("sin_chaleco", datetime(2026, 11, 20, 9, 40)),
               ev("con_casco", datetime(2026, 11, 20, 9, 50)),       # no es alerta
               ev("sin_casco", datetime(2026, 9, 1, 9, 0))]          # fuera del período
    m = riesgo.matriz(eventos, AHORA, dias=28)
    assert m[4][9] == 2 and sum(map(sum, m)) == 2
    assert riesgo.matriz(eventos, AHORA, tipo="sin_chaleco")[4][9] == 1


def test_puntos_de_una_camara_con_posicion():
    eventos = [ev("sin_casco", AHORA - timedelta(hours=1), pos=[0.2, 0.8]),
               ev("sin_casco", AHORA - timedelta(hours=1), camara="Otra", pos=[0.5, 0.5]),
               ev("zona_prohibida", AHORA - timedelta(hours=2))]       # sin posición: no se dibuja
    assert riesgo.puntos(eventos, "Portón", AHORA) == [[0.2, 0.8, "sin_casco"]]


def test_pronostico_pesa_mas_las_semanas_recientes():
    # Mañana es lunes 23/11. Lunes anteriores: 16/11 (peso 4) con 2 alertas a las 10 h, 9/11 (peso 3) con 0.
    eventos = [ev("sin_casco", datetime(2026, 11, 16, 10, 5)), ev("sin_casco", datetime(2026, 11, 16, 10, 30)),
               ev("sin_casco", datetime(2026, 11, 2, 14, 0))]          # 2/11: peso 2
    p = riesgo.pronostico(eventos, AHORA)
    assert p["dia"] == "lunes" and p["fecha"] == "2026-11-23"
    assert p["horas"][10] == round(2 * 4 / 10, 2) and p["horas"][14] == round(2 / 10, 2)
    assert [x["hora"] for x in p["picos"]] == [10, 14]


def test_tendencia_compara_semanas():
    eventos = [ev("sin_casco", AHORA - timedelta(days=1)), ev("sin_casco", AHORA - timedelta(days=2)),
               ev("sin_casco", AHORA - timedelta(days=9))]
    assert riesgo.tendencia(eventos, AHORA)["sin_casco"] == [2, 1]


def test_reporte_y_texto_del_dia():
    dia = datetime(2026, 11, 20)
    eventos = [ev("sin_casco", dia.replace(hour=9), persona="José"),
               ev("sin_casco", dia.replace(hour=9, minute=30), persona="José, Ana"),
               ev("hombre_caido", dia.replace(hour=15), camara="Galpón", persona="Sin identificar"),
               ev("sin_casco", dia - timedelta(days=7))]
    r = riesgo.reporte(eventos, dia.date())
    assert (r["total"], r["mismo_dia_semana_anterior"]) == (3, 1)
    assert r["reincidentes"] == [{"persona": "José", "alertas": 2}]
    assert r["hora_pico"] == {"hora": 9, "alertas": 2}
    t = riesgo.texto(r)
    assert "3 alertas, 2 más que el viernes anterior (1)" in t and "José (2)" in t and "09:00" in t


def test_reporte_sin_alertas():
    assert "Sin alertas" in riesgo.texto(riesgo.reporte([], datetime(2026, 11, 20).date()))


def test_redactar_sin_clave_usa_la_plantilla(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = riesgo.reporte([], datetime(2026, 11, 20).date())
    assert riesgo.redactar(r) == riesgo.texto(r)


def test_redactar_con_claude_y_respaldo_si_falla(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-de-prueba")
    r = riesgo.reporte([], datetime(2026, 11, 20).date())
    respuesta = mock.MagicMock(stop_reason="end_turn", content=[mock.MagicMock(type="text", text="Día tranquilo.")])
    with mock.patch.object(anthropic, "Anthropic") as cliente:
        cliente.return_value.beta.messages.create.return_value = respuesta
        assert riesgo.redactar(r) == "Día tranquilo."
        kwargs = cliente.return_value.beta.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-opus-5-5" and json.loads(kwargs["messages"][0]["content"].split("\n", 1)[1])["total"] == 0
        cliente.return_value.beta.messages.create.side_effect = RuntimeError("sin red")
        assert riesgo.redactar(r) == riesgo.texto(r)


def test_redactar_con_gemini_gratis_y_respaldo_si_falla(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "clave-gemini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "no-se-usa")   # con las dos, va Gemini (gratis)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    r = riesgo.reporte([], datetime(2026, 11, 20).date())
    ok = mock.MagicMock(status_code=200)
    ok.json.return_value = {"candidates": [{"content": {"parts": [{"text": "pensando", "thought": True},
                                                                  {"text": "Día tranquilo."}]}}]}
    with mock.patch("requests.post", return_value=ok) as post:
        assert riesgo.redactar(r) == "Día tranquilo."
        url, kw = post.call_args.args[0], post.call_args.kwargs
        assert url.endswith("/models/gemini-flash-latest:generateContent")
        assert kw["headers"] == {"x-goog-api-key": "clave-gemini"} and "clave" not in url
        assert json.loads(kw["json"]["contents"][0]["parts"][0]["text"].split("\n", 1)[1])["total"] == 0
        ok.json.return_value = {"candidates": []}            # bloqueado o vacío → plantilla
        assert riesgo.redactar(r) == riesgo.texto(r)
        ocupado = mock.MagicMock(status_code=503, headers={"content-type": "application/json"})
        ocupado.json.return_value = {"error": {"message": "high demand"}}
        ok.json.return_value = {"candidates": [{"content": {"parts": [{"text": "Con el de respaldo."}]}}]}
        post.side_effect = [ocupado, ok]                    # saturado → prueba el siguiente modelo
        assert riesgo.redactar(r) == "Con el de respaldo."
        assert post.call_args.args[0].endswith("/models/gemini-flash-lite-latest:generateContent")
        post.side_effect = None
        post.return_value = ocupado                         # todos saturados → plantilla
        assert riesgo.redactar(r) == riesgo.texto(r)
        post.side_effect = RuntimeError("cuota agotada")
        assert riesgo.redactar(r) == riesgo.texto(r)


# ── Telegram ─────────────────────────────────────────────────────────────────

@pytest.fixture
def telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100")
    monkeypatch.setattr(aviso_telegram, "_ultimo", {})
    hilos = []
    monkeypatch.setattr(aviso_telegram.threading, "Thread",
                        lambda target, args, daemon: hilos.append((target, args)) or mock.MagicMock())
    return hilos


def test_telegram_sin_configurar_no_hace_nada(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert aviso_telegram.avisar({"type": "sin_casco", "camera": "x"}) is False


def test_telegram_no_repite_la_misma_alerta_enseguida(telegram):
    e = {"type": "sin_casco", "camera": "Portón"}
    assert aviso_telegram.avisar(e) is True
    assert aviso_telegram.avisar(e) is False                               # misma cámara y tipo: espera
    assert aviso_telegram.avisar({"type": "zona_prohibida", "camera": "Portón"}) is True
    assert len(telegram) == 2


def test_telegram_junta_casco_y_chaleco_en_un_mensaje(telegram, monkeypatch):
    """Sin casco y sin chaleco de la misma cámara, con segundos de diferencia: un solo mensaje."""
    monkeypatch.setattr(aviso_telegram, "_pendientes", {})
    assert aviso_telegram.avisar({"type": "sin_casco", "camera": "Portón", "timestamp": "2026-11-22T18:52:46"}, b"a")
    assert aviso_telegram.avisar({"type": "sin_chaleco", "camera": "Portón", "timestamp": "2026-11-22T18:52:49"}, b"b")
    assert aviso_telegram.avisar({"type": "sin_chaleco", "camera": "Taller"})   # otra cámara: aparte
    assert len(telegram) == 2
    enviados = []
    monkeypatch.setattr(aviso_telegram, "_enviar", lambda e, f: enviados.append((e, f)))
    target, args = telegram[0]
    target(*args, espera=0)
    evento, foto = enviados[0]
    assert evento["tipos"] == ["sin_casco", "sin_chaleco"] and foto == b"b"
    assert aviso_telegram.leyenda(evento).startswith("⛑️🦺 SIN CASCO Y SIN CHALECO · Portón")


def test_telegram_manda_la_foto_en_jpeg(telegram):
    ok, webp = cv2.imencode(".webp", np.full((40, 60, 3), 90, np.uint8))
    e = {"type": "sin_casco", "camera": "Portón", "ubicacion": "Sector A", "persona": "José",
         "confidence": 91.2, "timestamp": "2026-11-22T09:15:30"}
    with mock.patch("aviso_telegram.requests.post", return_value=mock.MagicMock(status_code=200)) as post:
        aviso_telegram._enviar(e, webp.tobytes())
    url, = post.call_args.args
    assert url.endswith("/bot123:abc/sendPhoto")
    nombre, jpg, tipo = post.call_args.kwargs["files"]["photo"]
    assert tipo == "image/jpeg" and jpg[:2] == b"\xff\xd8"
    leyenda = post.call_args.kwargs["data"]["caption"]
    assert "SIN CASCO" in leyenda and "Sector A" in leyenda and "José" in leyenda and "09:15:30" in leyenda


def test_alerta_registrada_avisa_por_telegram(operator_client, monkeypatch):
    avisos = []
    monkeypatch.setattr(aviso_telegram, "avisar", lambda e, foto=None: avisos.append(e))
    with mock.patch("admin.enviar_comando_esp32"):
        operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80})
        operator_client.post("/admin/api/events", json={"type": "con_casco", "confidence": 80})
    assert [a["type"] for a in avisos] == ["sin_casco"]


# ── Posición de las alertas ──────────────────────────────────────────────────

@pytest.mark.parametrize("pos, guardada", [([0.25, 0.5], [0.25, 0.5]), ([1.5, 0.2], None), ("x", None),
                                           ([True, 0.2], None), ([0.1], None)])
def test_api_guarda_la_posicion_valida(operator_client, pos, guardada):
    with mock.patch("admin.enviar_comando_esp32"):
        operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80, "pos": pos})
    assert data_store.load_events()[0].get("pos") == guardada


def test_motor_guarda_donde_estaba_la_cabeza(monkeypatch):
    cam = data_store.create_camera("Portón", "", "usb", "0")
    eventos = []
    monkeypatch.setattr(motor, "registrar_evento", lambda *a: eventos.append(a))
    cara = {"bbox": [300, 120, 40, 40], "nombre": None, "con_casco": 5.0, "sin_casco": 95.0}
    monkeypatch.setattr(motor, "analizar", lambda img, modos, con_personas=False, **kw: ([cara], []))
    hilo = motor.Camara(cam)
    for _ in range(3):
        hilo._procesar(np.zeros((480, 640, 3), np.uint8))
    assert eventos[0][0] == "sin_casco" and eventos[0][5] == [320 / 640, 140 / 480]


# ── Páginas y reporte automático ─────────────────────────────────────────────

def test_paginas_de_riesgo_y_reporte(operator_client):
    data_store.append_event("sin_casco", 90, "Portón", pos=[0.3, 0.4])
    assert operator_client.get("/admin/riesgo").status_code == 200
    d = operator_client.get("/admin/api/riesgo?camara=Portón&dias=7&tipo=sin_casco").get_json()
    assert d["puntos"] == [[0.3, 0.4, "sin_casco"]] and len(d["matriz"]) == 7 and len(d["pronostico"]["horas"]) == 24
    hoy = datetime.now().date().isoformat()
    r = operator_client.get(f"/admin/reporte?fecha={hoy}")
    assert r.status_code == 200 and "1 alertas" in r.get_data(as_text=True)
    assert operator_client.get("/admin/reporte?fecha=basura").status_code == 200


def test_solo_admin_envia_el_reporte(operator_client, monkeypatch):
    enviados = []
    monkeypatch.setattr(aviso_telegram, "mandar_texto", lambda t: enviados.append(t) or True)
    operator_client.post("/admin/reporte/enviar", data={"texto": "hola"})
    assert enviados == []


def test_reporte_diario_se_manda_una_sola_vez(tmp_data, monkeypatch):
    enviados = []
    monkeypatch.setattr(aviso_telegram, "mandar_texto", lambda t: enviados.append(t) or True)
    dia = datetime(2026, 11, 20).date()
    assert admin.enviar_reporte_del_dia(dia) is True
    assert admin.enviar_reporte_del_dia(dia) is False                     # ya salió
    assert len(enviados) == 1 and "2026-11-20" in enviados[0]


def test_reporte_diario_que_no_salio_se_reintenta(tmp_data, monkeypatch):
    monkeypatch.setattr(aviso_telegram, "mandar_texto", lambda t: False)
    dia = datetime(2026, 11, 20).date()
    assert admin.enviar_reporte_del_dia(dia) is False
    assert not os.path.exists(os.path.join(data_store.DATA_DIR, "reporte-2026-11-20.enviado"))


def test_errores_de_telegram_no_muestran_el_token(telegram):
    err = ConnectionError("HTTPSConnectionPool: Max retries with url: /bot123:abc/sendPhoto")
    assert "123:abc" not in aviso_telegram.sin_token(err) and "***" in aviso_telegram.sin_token(err)


def test_reporte_diario_con_lock_viejo_igual_sale(tmp_data, monkeypatch):
    monkeypatch.setattr(aviso_telegram, "mandar_texto", lambda t: True)
    os.makedirs(data_store.DATA_DIR, exist_ok=True)
    lock = os.path.join(data_store.DATA_DIR, "reporte-2026-11-20.enviado.lock")
    open(lock, "w").close()
    os.utime(lock, (0, 0))                                                   # de un proceso que murió
    assert admin.enviar_reporte_del_dia(datetime(2026, 11, 20).date()) is True
    assert not os.path.exists(lock)


def test_kpis_por_turno_y_tiempo_de_correccion():
    eventos = [ev("sin_casco", datetime(2026, 11, 20, 9, 0)),
               ev("con_casco", datetime(2026, 11, 20, 9, 4)),            # corregida en 4 min
               ev("sin_chaleco", datetime(2026, 11, 20, 15, 0)),
               ev("con_casco", datetime(2026, 11, 20, 16, 0)),           # 60 min: no cuenta
               ev("hombre_caido", datetime(2026, 11, 20, 23, 30), camara="Galpón")]
    k = riesgo.kpis(eventos, AHORA, dias=7)
    t = {f["turno"]: f for f in k["turnos"]}
    assert (t["Mañana"]["alertas"], t["Mañana"]["correccion_min"], t["Mañana"]["corregidas"]) == (1, 4.0, 1)
    assert (t["Tarde"]["alertas"], t["Tarde"]["correccion_min"]) == (1, None)
    assert t["Noche"]["por_tipo"] == {"hombre_caido": 1}
    assert k["total"] == 3 and k["dias_sin_alertas"] == 2                # 21 y 22 sin alertas
    assert len(k["por_dia"]) == 7 and k["por_dia"][-1]["fecha"] == "2026-11-22"


def test_csv_de_kpis(operator_client):
    data_store.append_event("sin_casco", 90, "Portón")
    r = operator_client.get("/admin/riesgo/kpis.csv?dias=7")
    texto = r.get_data(as_text=True)
    assert r.status_code == 200 and "Turno,Alertas" in texto and "Fecha,Alertas" in texto
    assert operator_client.get("/admin/api/riesgo").get_json()["kpis"]["total"] == 1
