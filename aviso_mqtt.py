"""Eventos por MQTT, el estándar de automatización industrial (SCADA, PLC, Node-RED, Home Assistant).

Cada evento se publica como JSON en <MQTT_TOPIC>/<tipo> (ej. rac/eventos/sin_casco) con QoS 1, así
un PLC puede cortar un molinete o prender una baliza sin pasar por el panel.

Variables: MQTT_HOST (sin ella no hace nada), MQTT_PORT (1883, o 8883 con MQTT_TLS=1),
MQTT_USER / MQTT_PASS, MQTT_TOPIC (rac/eventos). La conexión es una sola, en segundo plano y con
reconexión automática de paho: si el broker se cae, RAC sigue funcionando y los avisos se pierden.
"""
import json
import os
import threading

_cliente = None
_lock = threading.Lock()


def configurado():
    return bool(os.environ.get("MQTT_HOST"))


def _conectar():
    import paho.mqtt.client as mqtt
    tls = os.environ.get("MQTT_TLS") == "1"
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"rac-vision-{os.getpid()}")
    if os.environ.get("MQTT_USER"):
        c.username_pw_set(os.environ["MQTT_USER"], os.environ.get("MQTT_PASS", ""))
    if tls:
        c.tls_set()
    c.reconnect_delay_set(min_delay=1, max_delay=60)
    c.max_queued_messages_set(1000)   # broker caído: no se acumulan eventos sin límite en memoria
    c.connect_async(os.environ["MQTT_HOST"], int(os.environ.get("MQTT_PORT", 8883 if tls else 1883)), keepalive=60)
    c.loop_start()
    return c


def publicar(evento):
    """Publica el evento (sin la foto) en <MQTT_TOPIC>/<tipo>. True si quedó encolado."""
    global _cliente
    if not configurado():
        return False
    try:
        with _lock:
            if _cliente is None:
                _cliente = _conectar()
        tema = f"{os.environ.get('MQTT_TOPIC', 'rac/eventos').rstrip('/')}/{evento['type']}"
        datos = {k: v for k, v in evento.items() if k != "foto"}
        return _cliente.publish(tema, json.dumps(datos, ensure_ascii=False), qos=1).rc == 0
    except Exception as e:  # broker caído o mal configurado: RAC sigue
        print(f"[!] MQTT no publicó el evento: {type(e).__name__}: {e}", flush=True)
        return False
