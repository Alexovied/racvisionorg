import json
import os
import secrets
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

# Un solo candado para leer y escribir los JSON: el motor de cámaras (motor.py) registra
# eventos desde sus hilos mientras el panel lee. Reentrante: append_event lo toma y adentro lee y guarda.
# shortcut: protege un solo proceso; con más de un worker de gunicorn, pasar los eventos a SQLite.
_lock = threading.RLock()

def _load_json(filename, default=None):
    _ensure_dir()
    path = os.path.join(DATA_DIR, filename)
    with _lock:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return default if default is not None else []

def _save_json(filename, data, indent=2):
    _ensure_dir()
    path = os.path.join(DATA_DIR, filename)
    with _lock:
        # Se escribe aparte y se reemplaza: un corte de luz a mitad no deja el archivo truncado
        # .tmp propio de cada proceso: dos workers escribiendo a la vez no mezclan sus archivos
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            # dumps y no dump: json.dump va siempre por el codificador lento en Python puro
            f.write(json.dumps(data, ensure_ascii=False, indent=indent))
            # Sin fsync el reemplazo puede quedar en disco antes que los datos (archivo vacío tras un corte)
            f.flush()
            os.fsync(f.fileno())
        # En Windows el reemplazo falla si otro programa (antivirus, backup, un editor) tiene
        # el archivo abierto en ese instante: se reintenta un cuarto de segundo antes de rendirse.
        for intento in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if intento == 4:
                    raise
                time.sleep(0.05)


# ═══════════════════════════════════════
#  USERS
# ═══════════════════════════════════════

def _default_admin():
    """Crea el admin inicial sin contraseña fija en el código.

    Usa ADMIN_PASSWORD (y opcionalmente ADMIN_USERNAME) si están definidas.
    Si no, genera una contraseña aleatoria y la muestra UNA sola vez en consola.
    Solo se llama cuando users.json no existe o está vacío.
    """
    username = os.environ.get("ADMIN_USERNAME", "").strip() or "admin"
    password = os.environ.get("ADMIN_PASSWORD", "")
    generated = not password
    if generated:
        password = secrets.token_urlsafe(16)

    if generated:
        print("=" * 60, file=sys.stderr)
        print(f"[!] Admin inicial creado: usuario '{username}'", file=sys.stderr)
        print(f"[!] Contraseña generada (se muestra solo esta vez): {password}", file=sys.stderr)
        print("[!] Guardala y cambiala desde el panel /admin.", file=sys.stderr)
        print("=" * 60, file=sys.stderr, flush=True)
    else:
        print(f"[+] Admin inicial '{username}' creado con ADMIN_PASSWORD.", file=sys.stderr)

    return [{
        "id": str(uuid.uuid4()),
        "username": username,
        "password_hash": generate_password_hash(password),
        "role": "admin",
        "created_at": datetime.now().isoformat()
    }]

def load_users():
    users = _load_json("users.json")
    if not users:
        users = _default_admin()
        _save_json("users.json", users)
    return users

def save_users(users):
    _save_json("users.json", users)

def get_user_by_id(user_id):
    for u in load_users():
        if u["id"] == user_id:
            return u
    return None

def get_user_by_username(username):
    for u in load_users():
        if u["username"] == username:
            return u
    return None

def create_user(username, password, role="operator"):
    users = load_users()
    # Check duplicates
    for u in users:
        if u["username"] == username:
            return None
    new_user = {
        "id": str(uuid.uuid4()),
        "username": username,
        "password_hash": generate_password_hash(password),
        "role": role,
        "created_at": datetime.now().isoformat()
    }
    users.append(new_user)
    save_users(users)
    return new_user

def update_user(user_id, username=None, password=None, role=None):
    users = load_users()
    for u in users:
        if u["id"] == user_id:
            if username:
                # Check duplicate username
                for other in users:
                    if other["id"] != user_id and other["username"] == username:
                        return False
                u["username"] = username
            if password:
                u["password_hash"] = generate_password_hash(password)
            if role:
                u["role"] = role
            save_users(users)
            return True
    return False

def delete_user(user_id):
    """True si el usuario existía y se borró."""
    users = load_users()
    quedan = [u for u in users if u["id"] != user_id]
    save_users(quedan)
    return len(quedan) < len(users)

def verify_password(user, password):
    return check_password_hash(user["password_hash"], password)


# ═══════════════════════════════════════
#  CAMERAS
# ═══════════════════════════════════════

def load_cameras():
    return _load_json("cameras.json", [])

def save_cameras(cameras):
    _save_json("cameras.json", cameras)

def create_camera(name, location, cam_type="webcam", url="", **extra):
    """extra: campos propios del tipo (token_hash de las relay)."""
    cameras = load_cameras()
    new_cam = {**extra,
        "id": str(uuid.uuid4()),
        "name": name,
        "location": location,
        "type": cam_type,
        "url": url,
        "active": True,
        "created_at": datetime.now().isoformat()
    }
    cameras.append(new_cam)
    save_cameras(cameras)
    return new_cam

def toggle_camera(cam_id):
    cameras = load_cameras()
    for c in cameras:
        if c["id"] == cam_id:
            c["active"] = not c["active"]
            save_cameras(cameras)
            return c
    return None

def delete_camera(cam_id):
    """True si la cámara existía y se borró."""
    cameras = load_cameras()
    quedan = [c for c in cameras if c["id"] != cam_id]
    save_cameras(quedan)
    return len(quedan) < len(cameras)

def set_zonas(cam_id, zonas, exentas=None, horario=False):
    """Zonas de exclusión de una cámara: polígonos [[x, y], ...] con coordenadas de 0 a 1.
    exentas: polígonos donde no se exige EPP (None = no cambian). horario: {"desde", "hasta"} en que
    se exige EPP, None = siempre (False = no cambia)."""
    with _lock:
        cameras = load_cameras()
        for c in cameras:
            if c["id"] == cam_id:
                c["zonas"] = zonas
                if exentas is not None:
                    c["exentas"] = exentas
                if horario is not False:
                    c["horario"] = horario
                save_cameras(cameras)
                return c
    return None


# ═══════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════

DEFAULT_CONFIG = {
    "min_confidence": 51,
    "confirm_threshold": 3,
    "scan_rate_ms": 100,
    "active_model": "modelo.keras",
    "detectar_casco": True,
    "detectar_chaleco": True,
    # Solo para cámaras que analiza el servidor (motor.py)
    "detectar_caida": True,
    "caida_segundos": 10,
    # Modo anónimo: sin reconocimiento facial y con las caras pixeladas en las fotos de las alertas
    "modo_anonimo": False,
}

def load_config():
    config = _load_json("config.json", None)
    if not config or not isinstance(config, dict):
        config = DEFAULT_CONFIG.copy()
        _save_json("config.json", config)
    # Configs guardadas antes de que existieran claves nuevas (p. ej. detectar_chaleco)
    return {**DEFAULT_CONFIG, **config}

def save_config(config):
    _save_json("config.json", config)

def reset_config():
    _save_json("config.json", DEFAULT_CONFIG.copy())
    return DEFAULT_CONFIG.copy()


# ═══════════════════════════════════════
#  EVENTS / AUDIT LOG
# ═══════════════════════════════════════

# Infracciones: guardan persona y foto, disparan la alarma y cuentan como alerta en el panel
ALERT_TYPES = ("sin_casco", "sin_chaleco", "zona_prohibida", "hombre_caido",
               # de los wearables (wearables.py)
               "boton_sos", "hombre_muerto", "posible_desmayo", "ritmo_cardiaco", "wearable_sin_senal")
# Salud de las cámaras (motor.py + salud.py): avisan y guardan foto, pero no hacen sonar la alarma
SALUD_TYPES = ("camara_sin_senal", "camara_tapada", "camara_desenfocada")

def load_events():
    with _lock:
        try:
            return _load_json("events.json", [])
        except ValueError as e:
            # events.json dañado: se aparta (no se borra) y se sigue con un historial nuevo; si no,
            # ninguna alerta se registra ni se avisa hasta que alguien arregle el archivo a mano
            ruta = os.path.join(DATA_DIR, "events.json")
            aparte = f"events.json.corrupto-{int(time.time())}"
            os.replace(ruta, os.path.join(DATA_DIR, aparte))
            print(f"[!] events.json estaba dañado ({e}): quedó apartado como {aparte} y el historial arranca vacío",
                  file=sys.stderr, flush=True)
            return []

def alert_photo_path(event_id, ext="webp"):
    """Ruta de la foto de una alerta: data/alertas/<id>.webp
    (las alertas anteriores al cambio a WebP quedaron como <id>.jpg)."""
    return os.path.join(DATA_DIR, "alertas", f"{event_id}.{ext}")

def existing_alert_photo(event_id):
    """(ruta, mimetype) de la foto guardada de una alerta, WebP o JPEG viejo; None si no hay."""
    for ext, mime in (("webp", "image/webp"), ("jpg", "image/jpeg")):
        ruta = alert_photo_path(event_id, ext)
        if os.path.exists(ruta):
            return ruta, mime
    return None

def _remove_alert_photo(event_id):
    for ext in ("webp", "jpg"):
        try:
            os.remove(alert_photo_path(event_id, ext))
        except OSError:
            pass

def append_event(event_type, confidence, camera="webcam", foto_webp=None, **extra):
    """extra: persona, ubicacion... foto_webp: bytes WebP ya validados (se guarda aparte)."""
    with _lock:
        events = load_events()
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "confidence": round(confidence, 1),
            "camera": camera,
            "timestamp": datetime.now().isoformat(),
            **extra,
        }
        events.insert(0, event)
        # Keep max 5000 events (y se borran las fotos de los que salen)
        if len(events) > 5000:
            for viejo in events[5000:]:
                if viejo.get("foto"):
                    _remove_alert_photo(viejo["id"])
            events = events[:5000]
        try:
            if foto_webp:
                os.makedirs(os.path.dirname(alert_photo_path(event["id"])), exist_ok=True)
                with open(alert_photo_path(event["id"]), "wb") as f:
                    f.write(foto_webp)
                event["foto"] = True
            # sin sangría: events.json se reescribe entero en cada evento, con el candado tomado
            _save_json("events.json", events, indent=None)
        except Exception:
            if foto_webp:  # el evento no quedó registrado: su foto (entera o a medias) tampoco
                _remove_alert_photo(event["id"])
            raise
        return event

def get_event(event_id):
    for e in load_events():
        if e.get("id") == event_id:
            return e
    return None

def update_event(event_id, **campos):
    """Agrega o cambia campos de un evento guardado. Devuelve el evento o None si no existe."""
    with _lock:
        events = load_events()
        for e in events:
            if e.get("id") == event_id:
                e.update(campos)
                _save_json("events.json", events, indent=None)
                return e
    return None

def set_feedback(event_id, veredicto, usuario):
    """Marca una alerta como "real" o "falsa" (lo usa aprendizaje.py)."""
    return update_event(event_id, feedback={"veredicto": veredicto, "usuario": usuario,
                                            "fecha": datetime.now().isoformat(timespec="seconds")})

def get_filtered_events(event_type=None, camera=None, date_from=None, date_to=None, page=1, per_page=20):
    events = load_events()
    filtered = []
    for e in events:
        if event_type and event_type != "all" and e.get("type") != event_type:
            continue
        if camera and camera != "all" and e.get("camera") != camera:
            continue
        if date_from:
            try:
                evt_date = datetime.fromisoformat(e["timestamp"]).date()
                from_date = datetime.strptime(date_from, "%Y-%m-%d").date()
                if evt_date < from_date:
                    continue
            except (ValueError, KeyError):
                pass
        if date_to:
            try:
                evt_date = datetime.fromisoformat(e["timestamp"]).date()
                to_date = datetime.strptime(date_to, "%Y-%m-%d").date()
                if evt_date > to_date:
                    continue
            except (ValueError, KeyError):
                pass
        filtered.append(e)
    
    total = len(filtered)
    if page < 1 or per_page < 1:   # filtered[-25:0] daría los últimos eventos
        return [], total
    start = (page - 1) * per_page
    end = start + per_page
    return filtered[start:end], total


# ═══════════════════════════════════════
#  STATISTICS
# ═══════════════════════════════════════

def get_stats():
    events = load_events()
    cameras = load_cameras()
    users = load_users()
    
    now = datetime.now()
    last_24h = now - timedelta(hours=24)
    
    total_events = len(events)
    alerts_24h = 0
    detections_by_hour = {}
    
    for e in events:
        try:
            ts = datetime.fromisoformat(e["timestamp"])
            if ts >= last_24h:
                if e.get("type") in ALERT_TYPES:
                    alerts_24h += 1
                hour_key = ts.strftime("%H:00")
                if hour_key not in detections_by_hour:
                    detections_by_hour[hour_key] = {"con_casco": 0, "sin_casco": 0}
                evt_type = e.get("type", "con_casco")
                if evt_type in ALERT_TYPES:
                    evt_type = "sin_casco"   # la barra roja junta todas las alertas, como la tarjeta "Alertas 24 h"
                if evt_type in detections_by_hour[hour_key]:
                    detections_by_hour[hour_key][evt_type] += 1
        except (ValueError, KeyError):
            pass
    
    # Build ordered hours for chart
    hours_labels = []
    con_casco_data = []
    sin_casco_data = []
    for i in range(24):
        h = (now - timedelta(hours=23 - i)).strftime("%H:00")
        hours_labels.append(h)
        d = detections_by_hour.get(h, {"con_casco": 0, "sin_casco": 0})
        con_casco_data.append(d["con_casco"])
        sin_casco_data.append(d["sin_casco"])
    
    active_cameras = sum(1 for c in cameras if c.get("active", False))
    recent_events = events[:10]
    
    return {
        "total_events": total_events,
        "alerts_24h": alerts_24h,
        "active_cameras": active_cameras,
        "total_cameras": len(cameras),
        "total_users": len(users),
        "hours_labels": hours_labels,
        "con_casco_data": con_casco_data,
        "sin_casco_data": sin_casco_data,
        "recent_events": recent_events
    }
