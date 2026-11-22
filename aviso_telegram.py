"""Alertas por Telegram: la foto del momento con quién, dónde y qué faltó, al celular del supervisor.

Se activa con dos variables de entorno: TELEGRAM_BOT_TOKEN (del @BotFather) y TELEGRAM_CHAT_ID
(el chat o grupo que recibe). Sin ellas no hace nada. Nunca frena a quien la llama: envía en
un hilo aparte, y una misma cámara con el mismo tipo de alerta avisa como mucho cada ESPERA segundos
(si no, una persona sin casco frente a la cámara llena el chat).
"""
import os
import threading
import time

import cv2
import numpy as np
import requests

ESPERA = 60
TITULOS = {"sin_casco": "⛑️ SIN CASCO", "sin_chaleco": "🦺 SIN CHALECO",
           "zona_prohibida": "🚫 ZONA PROHIBIDA", "hombre_caido": "🆘 HOMBRE CAÍDO",
           "boton_sos": "🚨 BOTÓN DE PÁNICO", "hombre_muerto": "🆘 HOMBRE MUERTO SIN RESPUESTA",
           "posible_desmayo": "🆘 POSIBLE DESMAYO", "ritmo_cardiaco": "❤️ RITMO CARDÍACO",
           "wearable_sin_senal": "📡 WEARABLE SIN SEÑAL",
           "camara_sin_senal": "📵 CÁMARA SIN SEÑAL", "camara_tapada": "🙈 CÁMARA TAPADA",
           "camara_desenfocada": "🌫️ CÁMARA DESENFOCADA"}
_ultimo = {}
_lock = threading.Lock()


def configurado():
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def sin_token(e):
    """Texto de un error sin el token: los errores de requests traen la URL completa (bot<TOKEN>)."""
    texto, token = str(e), os.environ.get("TELEGRAM_BOT_TOKEN", "")
    return texto.replace(token, "***") if token else texto


def _url(metodo):
    return f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/{metodo}"


def leyenda(evento):
    t = evento.get("timestamp", "")[11:19]
    partes = [f"{TITULOS.get(evento['type'], evento['type'])} · {evento.get('camera', '')}"]
    if evento.get("ubicacion"):
        partes.append(f"📍 {evento['ubicacion']}")
    if evento.get("detalle"):
        partes.append(f"ℹ️ {evento['detalle']}")
    if evento.get("confirmado_por"):
        partes.append(f"✅ Confirmado por {evento['confirmado_por']}")
    if evento["type"].startswith("camara_"):
        partes.append(f"Punto ciego desde las {t}: revisar la cámara")
    else:
        partes.append(f"👤 {evento.get('persona') or 'Sin identificar'} · {float(evento.get('confidence') or 0):.0f}% · {t}")
    return "\n".join(partes)


def _enviar(evento, foto):
    try:
        datos = {"chat_id": os.environ["TELEGRAM_CHAT_ID"], "caption": leyenda(evento)}
        jpg = None
        if foto:  # Telegram muestra mejor JPEG que WebP como foto
            img = cv2.imdecode(np.frombuffer(foto, np.uint8), cv2.IMREAD_COLOR)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85]) if img is not None else (False, None)
            jpg = buf.tobytes() if ok else None
        if jpg:
            r = requests.post(_url("sendPhoto"), data=datos, files={"photo": ("alerta.jpg", jpg, "image/jpeg")},
                              timeout=10)
        else:
            r = requests.post(_url("sendMessage"), data={"chat_id": datos["chat_id"], "text": datos["caption"]},
                              timeout=10)
        if r.status_code >= 400:
            print(f"[!] Telegram respondió {r.status_code}: {r.text[:200]}", flush=True)
    except Exception as e:
        print(f"[!] Telegram no recibió la alerta: {sin_token(e)}", flush=True)


def avisar(evento, foto=None):
    """Manda la alerta en segundo plano. True si salió a enviarse; False si no está configurado
    o si la misma cámara ya avisó ese tipo hace menos de ESPERA segundos."""
    if not configurado():
        return False
    clave = (evento.get("camera"), evento.get("type"))
    ahora = time.monotonic()
    with _lock:
        if ahora - _ultimo.get(clave, -ESPERA) < ESPERA:
            return False
        _ultimo[clave] = ahora
    threading.Thread(target=_enviar, args=(evento, foto), daemon=True).start()
    return True


def mandar_texto(texto):
    """Mensaje de texto (reporte diario). Devuelve True si Telegram lo aceptó."""
    if not configurado():
        return False
    try:
        r = requests.post(_url("sendMessage"), data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": texto[:4000]},
                          timeout=15)
        if r.status_code >= 400:
            print(f"[!] Telegram respondió {r.status_code} al reporte: {r.text[:200]}", flush=True)
        return r.status_code < 400
    except requests.RequestException as e:
        print(f"[!] Telegram no recibió el reporte: {sin_token(e)}", flush=True)
        return False
