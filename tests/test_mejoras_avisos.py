"""Avisos que no se pierden: reintentos de Telegram, reconexión del ESP32 y log de MQTT."""
from unittest import mock

import pytest
import requests

import aviso_mqtt
import aviso_telegram
import hardware


# ── Telegram ─────────────────────────────────────────────────────────────────

@pytest.fixture
def telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100")
    monkeypatch.setattr(aviso_telegram, "_ultimo", {})
    esperas = []
    monkeypatch.setattr(aviso_telegram.time, "sleep", esperas.append)
    return esperas


EVENTO = {"type": "hombre_caido", "camera": "Portón", "timestamp": "2026-11-26T09:15:30"}


def test_telegram_reintenta_tras_un_corte_de_red(telegram):
    respuestas = [requests.ConnectionError("sin red"), mock.MagicMock(status_code=502, text="Bad Gateway"),
                  mock.MagicMock(status_code=200)]
    with mock.patch("aviso_telegram.requests.post", side_effect=respuestas) as post:
        aviso_telegram._enviar(EVENTO, None)
    assert post.call_count == 3 and telegram == [5, 10]


def test_telegram_si_rechaza_la_foto_manda_el_texto(telegram):
    import cv2
    import numpy as np
    ok, webp = cv2.imencode(".webp", np.full((40, 60, 3), 90, np.uint8))
    respuestas = [mock.MagicMock(status_code=400, text="IMAGE_PROCESS_FAILED"), mock.MagicMock(status_code=200)]
    with mock.patch("aviso_telegram.requests.post", side_effect=respuestas) as post:
        aviso_telegram._enviar(EVENTO, webp.tobytes())
    assert [c.args[0].rsplit("/", 1)[1] for c in post.call_args_list] == ["sendPhoto", "sendMessage"]
    assert "HOMBRE CAÍDO" in post.call_args.kwargs["data"]["text"] and telegram == []


def test_telegram_con_el_token_mal_no_insiste(telegram):
    with mock.patch("aviso_telegram.requests.post",
                    return_value=mock.MagicMock(status_code=401, text="Unauthorized")) as post:
        aviso_telegram._enviar(EVENTO, None)
    assert post.call_count == 1 and telegram == []


def test_telegram_si_no_salio_la_proxima_alerta_no_se_descarta(telegram, monkeypatch):
    monkeypatch.setattr(aviso_telegram.threading, "Thread", lambda target, args, daemon: mock.MagicMock())
    assert aviso_telegram.avisar(EVENTO) is True
    assert aviso_telegram.avisar(EVENTO) is False          # la espera normal de ESPERA segundos
    with mock.patch("aviso_telegram.requests.post", side_effect=requests.ConnectionError("sin red")) as post:
        aviso_telegram._enviar(EVENTO, None)
    assert post.call_count == aviso_telegram.INTENTOS
    assert aviso_telegram.avisar(EVENTO) is True           # no llegó: la siguiente sale igual


def test_telegram_sin_respuesta_no_duplica_ni_libera_la_espera(telegram, monkeypatch):
    monkeypatch.setattr(aviso_telegram.threading, "Thread", lambda target, args, daemon: mock.MagicMock())
    assert aviso_telegram.avisar(EVENTO) is True
    with mock.patch("aviso_telegram.requests.post", side_effect=requests.ReadTimeout("sin respuesta")) as post:
        aviso_telegram._enviar(EVENTO, None)
    assert post.call_count == 1 and telegram == []         # pudo haberse publicado: no se repite
    assert aviso_telegram.avisar(EVENTO) is False          # y la espera de ESPERA segundos sigue en pie


# ── ESP32 ────────────────────────────────────────────────────────────────────

class _Puerto:
    in_waiting = 0
    is_open = True
    port = "COM-TEST"

    def __init__(self, falla=False):
        self.escritos, self.falla = [], falla

    def write(self, b):
        if self.falla:
            raise OSError("cable desenchufado")
        self.escritos.append(b)

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture
def esp32(monkeypatch):
    """ESP32 conectado que falla al escribir; el hilo de reconexión corre en el acto y sin esperar."""
    monkeypatch.setattr(hardware, "esp32", _Puerto(falla=True))
    monkeypatch.setattr(hardware, "_enabled", True)
    monkeypatch.setattr(hardware, "_reintento", None)
    monkeypatch.setattr(hardware.time, "sleep", lambda s: None)
    hilo = lambda target, args, daemon: mock.MagicMock(start=lambda: target(*args))
    monkeypatch.setattr(hardware.threading, "Thread", hilo)
    nuevo = _Puerto()
    monkeypatch.setattr(hardware.serial, "Serial", mock.MagicMock(return_value=nuevo))
    return nuevo


def test_esp32_se_reconecta_despues_de_un_error_de_escritura(esp32, monkeypatch):
    assert hardware.enviar_comando_esp32("R") is False
    assert hardware.get_status()["connected"] is False
    assert hardware.enviar_comando_esp32("R") is False and not hardware.serial.Serial.called   # todavía no toca
    monkeypatch.setattr(hardware, "_reintento", hardware._reintento - hardware.REINTENTO - 1)
    assert hardware.enviar_comando_esp32("R") is False      # este se pierde, pero dispara la reconexión
    assert hardware.serial.Serial.call_args.kwargs["write_timeout"] == 1
    assert hardware.enviar_comando_esp32("R") is True and esp32.escritos == [b"R"]


def test_esp32_que_nunca_conecto_no_reintenta(esp32, monkeypatch):
    monkeypatch.setattr(hardware, "_enabled", False)
    monkeypatch.setattr(hardware, "esp32", None)
    assert hardware.enviar_comando_esp32("R") is False and not hardware.serial.Serial.called


# ── MQTT ─────────────────────────────────────────────────────────────────────

def test_mqtt_avisa_en_el_log_si_el_broker_rechaza_o_no_responde(monkeypatch, capsys):
    monkeypatch.setenv("MQTT_HOST", "broker.local")
    monkeypatch.setattr(aviso_mqtt, "_ultimo_log", None)
    import paho.mqtt.client as mqtt
    with mock.patch.object(mqtt, "Client") as Client:
        c = aviso_mqtt._conectar()
    c.on_connect_fail(c, None)
    c.on_connect_fail(c, None)                              # paho reintenta sin parar: una sola línea
    c.on_connect(c, None, None, mock.MagicMock(is_failure=True, __str__=lambda s: "Not authorized"), None)
    c.on_connect(c, None, None, mock.MagicMock(is_failure=False), None)
    c.on_disconnect(c, None, None, "Keep alive timeout", None)
    lineas = capsys.readouterr().out.splitlines()
    assert len(lineas) == 4 and "no se pudo conectar" in lineas[0] and "Not authorized" in lineas[1]
    assert lineas[2] == "[+] MQTT conectado" and "Keep alive timeout" in lineas[3]


@pytest.mark.parametrize("rc, avisa", [(0, False), (4, False), (15, True)])
def test_mqtt_avisa_si_la_cola_llena_descarta_el_evento(monkeypatch, capsys, rc, avisa):
    monkeypatch.setenv("MQTT_HOST", "broker.local")
    cliente = mock.MagicMock()
    cliente.publish.return_value.rc = rc
    monkeypatch.setattr(aviso_mqtt, "_cliente", cliente)
    assert aviso_mqtt.publicar({"type": "sin_casco"}) is (rc == 0)
    assert ("descartó el evento sin_casco" in capsys.readouterr().out) is avisa
