"""
hardware.py — Canal serial centralizado para el ESP32.

Un único objeto serial compartido con threading.Lock() garantiza que
app.py y admin.py nunca colisionen al escribir al mismo tiempo.
Degradación graceful: si el puerto no existe el sistema sigue funcionando.
"""

import serial
import threading
import time
import os

# ─── Configuración via variables de entorno ────────────────────────────────────
# Windows: "COM3", "COM4"…   Linux/Mac: "/dev/ttyUSB0", "/dev/ttyACM0"
DEFAULT_PORT = os.environ.get("SERIAL_PORT", "COM4")
BAUD_RATE    = int(os.environ.get("SERIAL_BAUD", "115200"))

# ─── Estado interno (singleton) ────────────────────────────────────────────────
esp32    = None
_lock    = threading.Lock()
_enabled = False


def init_esp32(port=DEFAULT_PORT, baud=BAUD_RATE):
    """
    Inicializa (o reinicializa) la conexión serial con el ESP32.
    Retorna True si conectó exitosamente.
    """
    global esp32, _enabled

    with _lock:
        # Cerrar conexión previa si existe
        if esp32 is not None:
            try:
                if esp32.is_open:
                    esp32.close()
            except Exception:
                pass

        try:
            esp32 = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)          # Esperar reinicio del ESP32 tras DTR
            while esp32.in_waiting:
                esp32.readline()   # Limpiar buffer de arranque
            _enabled = True
            print(f"[+] ESP32 conectado y listo en {port} @ {baud} baud (hardware module).")
            return True
        except Exception as e:
            esp32    = None
            _enabled = False
            print(f"[!] ESP32 no conectado en {port}: {e}")
            print("    El servidor y panel web continuarán funcionando con normalidad.")
            return False


def enviar_comando_esp32(cmd):
    """
    Envía un comando al ESP32.

    Comandos del firmware:
        'R' / b'R'  -> LED Rojo  + Buzzer ON   (sin casco)
        'G' / b'G'  -> LED Verde + Buzzer OFF  (con casco / seguro)
        'O' / b'O'  -> Apagar todo (reposo)

    Acepta str o bytes para compatibilidad con ambos estilos de llamada.
    Retorna True si el envío fue exitoso.
    """
    global esp32, _enabled

    if not _enabled or esp32 is None:
        return False

    # Normalizar a bytes
    if isinstance(cmd, str):
        cmd_bytes = cmd.encode("utf-8")
    else:
        cmd_bytes = bytes(cmd)

    with _lock:
        try:
            esp32.write(cmd_bytes)
            esp32.flush()
            labels = {b'R': 'ALERTA SIN CASCO', b'G': 'ACCESO SEGURO CON CASCO', b'O': 'REPOSO'}
            label = labels.get(cmd_bytes, str(cmd_bytes))
            print(f"[ESP32] >>> {cmd_bytes} [{label}] enviado.")
            return True
        except Exception as e:
            print(f"[!] Error comunicando con ESP32: {e}")
            _enabled = False
            try:
                esp32.close()
            except Exception:
                pass
            esp32 = None
            return False


def get_status():
    """Estado de la conexión para diagnóstico."""
    return {
        "connected": _enabled,
        "port":      (esp32.port if esp32 else DEFAULT_PORT),
        "baud":      BAUD_RATE
    }
