"""Eventos por MQTT para SCADA / PLC."""
import json
from unittest import mock

import pytest

import aviso_mqtt
import data_store


@pytest.fixture
def broker(monkeypatch):
    monkeypatch.setenv("MQTT_HOST", "broker.local")
    monkeypatch.setenv("MQTT_TOPIC", "obra/rac/")
    cliente = mock.MagicMock()
    cliente.publish.return_value.rc = 0
    monkeypatch.setattr(aviso_mqtt, "_cliente", None)
    monkeypatch.setattr(aviso_mqtt, "_conectar", lambda: cliente)
    return cliente


def test_sin_host_no_publica(monkeypatch):
    monkeypatch.delenv("MQTT_HOST", raising=False)
    assert aviso_mqtt.publicar({"type": "sin_casco"}) is False


def test_publica_json_en_el_tema_del_tipo(broker):
    assert aviso_mqtt.publicar({"type": "sin_casco", "camera": "Portón", "foto": True}) is True
    tema, carga = broker.publish.call_args.args
    assert tema == "obra/rac/sin_casco" and broker.publish.call_args.kwargs["qos"] == 1
    assert json.loads(carga) == {"type": "sin_casco", "camera": "Portón"}


def test_broker_caido_no_rompe(broker):
    broker.publish.side_effect = OSError("conexión rechazada")
    assert aviso_mqtt.publicar({"type": "sin_casco"}) is False


def test_cada_evento_registrado_sale_por_mqtt(operator_client, broker):
    with mock.patch("admin.enviar_comando_esp32"):
        operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80})
        operator_client.post("/admin/api/events", json={"type": "con_casco", "confidence": 80})
    assert [c.args[0] for c in broker.publish.call_args_list] == ["obra/rac/sin_casco", "obra/rac/con_casco"]
    assert len(data_store.load_events()) == 2


def test_conectar_usa_credenciales_y_tls(monkeypatch):
    monkeypatch.setenv("MQTT_HOST", "broker.local")
    monkeypatch.setenv("MQTT_USER", "rac")
    monkeypatch.setenv("MQTT_PASS", "secreto")
    monkeypatch.setenv("MQTT_TLS", "1")
    import paho.mqtt.client as mqtt
    with mock.patch.object(mqtt, "Client") as Client:
        aviso_mqtt._conectar()
    c = Client.return_value
    c.username_pw_set.assert_called_once_with("rac", "secreto")
    c.tls_set.assert_called_once()
    c.connect_async.assert_called_once_with("broker.local", 8883, keepalive=60)
    c.loop_start.assert_called_once()
