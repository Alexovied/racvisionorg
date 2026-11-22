import glob
import csv
import io
import math
import os
import re
import threading
import time
import requests
from functools import wraps
from urllib.parse import urlsplit
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    jsonify, Response, send_file, abort
)
import base64
import binascii
import cv2
import numpy as np
from werkzeug.exceptions import HTTPException
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
import data_store
import face_id
import motor
from hardware import enviar_comando_esp32

# ── Integración n8n ──────────────────────────────────────────────────────────
# Configurar en Render (o .env local):
#   N8N_WEBHOOK_URL   → Production URL del Webhook de n8n (obligatorio)
#   N8N_WEBHOOK_TOKEN → Bearer token si el nodo Webhook de n8n usa Header Auth
#                       (opcional — dejar vacío si el webhook es público)
N8N_WEBHOOK_URL   = os.environ.get("N8N_WEBHOOK_URL", "")
N8N_WEBHOOK_TOKEN = os.environ.get("N8N_WEBHOOK_TOKEN", "")
# ─────────────────────────────────────────────────────────────────────────────

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# ═══════════════════════════════════════
#  USER MODEL for Flask-Login
# ═══════════════════════════════════════

class User(UserMixin):
    def __init__(self, user_dict):
        self.id = user_dict["id"]
        self.username = user_dict["username"]
        self.role = user_dict["role"]
        self._data = user_dict

def init_login(app):
    login_manager = LoginManager()
    login_manager.login_view = "admin.login"
    login_manager.login_message = "Inicia sesión para acceder al panel."
    login_manager.login_message_category = "warning"
    login_manager.init_app(app)

    @login_manager.unauthorized_handler
    def unauthorized():
        # La API responde 401 en JSON (los fetch no deben recibir HTML de login);
        # las páginas redirigen al login conservando a dónde se quería ir.
        if is_api_request():
            return jsonify({"error": "No autenticado"}), 401
        flash(login_manager.login_message, login_manager.login_message_category)
        return redirect(url_for("admin.login", next=request.full_path.rstrip("?")))

    @login_manager.user_loader
    def load_user(user_id):
        u = data_store.get_user_by_id(user_id)
        if u:
            return User(u)
        return None


# ═══════════════════════════════════════
#  ROLE DECORATOR
# ═══════════════════════════════════════

API_PREFIXES = ("/predict", "/reset_buffer", "/api/", "/admin/api/", "/sgep/api/")


def is_api_request():
    return request.path.startswith(API_PREFIXES)


def roles_required(*roles):
    """Exige sesión y uno de los roles indicados (403 si el rol no alcanza)."""
    def wrapper(f):
        @wraps(f)
        @login_required
        def decorated(*args, **kwargs):
            if current_user.role not in roles:
                if is_api_request():
                    return jsonify({"error": "Permisos insuficientes"}), 403
                flash("No tienes permisos para acceder a esta sección.", "error")
                return redirect(url_for("admin.dashboard"))
            return f(*args, **kwargs)
        return decorated
    return wrapper


def _safe_next(target):
    """Solo acepta rutas relativas del propio sitio (evita open redirect)."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc:
        return None
    return target


# pm y auditor son roles del SGEP (sgep.py): no acceden a nada de RAC
ROLES = ("admin", "operator", "viewer", "pm", "auditor")


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("No tienes permisos para acceder a esta sección.", "error")
            return redirect(url_for("admin.dashboard"))
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════

@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("admin.dashboard"))
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        user_data = data_store.get_user_by_username(username)
        if user_data and data_store.verify_password(user_data, password):
            user = User(user_data)
            login_user(user)
            next_page = _safe_next(request.args.get("next"))
            return redirect(next_page or url_for("admin.dashboard"))
        else:
            flash("Usuario o contraseña incorrectos.", "error")
    
    return render_template("admin/login.html")

@admin_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("admin.login"))


# ═══════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════

@admin_bp.route("/")
@login_required
def dashboard():
    stats = data_store.get_stats()
    return render_template("admin/dashboard.html", stats=stats)


# ═══════════════════════════════════════
#  USERS MANAGEMENT
# ═══════════════════════════════════════

@admin_bp.route("/users", methods=["GET", "POST"])
@login_required
@admin_required
def users():
    if request.method == "POST":
        action = request.form.get("action")
        
        if action == "create":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            role = request.form.get("role", "operator")
            
            if role not in ROLES:
                flash("Rol inválido.", "error")
            elif not username or not password:
                flash("Usuario y contraseña son obligatorios.", "error")
            elif len(password) < 4:
                flash("La contraseña debe tener al menos 4 caracteres.", "error")
            else:
                result = data_store.create_user(username, password, role)
                if result:
                    flash(f"Usuario '{username}' creado exitosamente.", "success")
                else:
                    flash(f"El usuario '{username}' ya existe.", "error")
        
        elif action == "edit":
            user_id = request.form.get("user_id")
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            role = request.form.get("role")
            if role and role not in ROLES:
                flash("Rol inválido.", "error")
                return redirect(url_for("admin.users"))
            
            success = data_store.update_user(
                user_id,
                username=username or None,
                password=password if password else None,
                role=role or None
            )
            if success:
                flash("Usuario actualizado.", "success")
            else:
                flash("Error: el nombre de usuario ya está en uso.", "error")
        
        return redirect(url_for("admin.users"))
    
    all_users = data_store.load_users()
    return render_template("admin/users.html", users=all_users)

@admin_bp.route("/users/<user_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    if user_id == current_user.id:
        flash("No puedes eliminar tu propio usuario.", "error")
    else:
        data_store.delete_user(user_id)
        flash("Usuario eliminado.", "success")
    return redirect(url_for("admin.users"))


# ═══════════════════════════════════════
#  PERSONAS (reconocimiento facial)
# ═══════════════════════════════════════

MAX_FOTOS = 10


def _nombre_valido(nombre):
    # Solo letras (con tildes), espacios y . ' - como en "María José O'Neil"; empieza con letra, sin números
    return 2 <= len(nombre) <= 60 and re.fullmatch(r"[^\W\d_](?:[^\W\d_]|[ .'-])*", nombre) is not None


@admin_bp.route("/personas", methods=["GET", "POST"])
@login_required
@admin_required
def personas():
    if request.method == "POST":
        nombre = " ".join(request.form.get("nombre", "").split())
        fotos = request.files.getlist("fotos")
        if not face_id.disponible():
            flash("El reconocimiento facial no está disponible en este servidor.", "error")
        elif not _nombre_valido(nombre):
            flash("Ingresá un nombre válido (2 a 60 letras, sin números).", "error")
        elif request.form.get("consentimiento") != "si":
            flash("Falta confirmar el consentimiento escrito y firmado de la persona.", "error")
        elif any(p["nombre"].lower() == nombre.lower() for p in face_id.listar()):
            flash(f"'{nombre}' ya está registrada. Para cambiar sus fotos, borrala y registrala de nuevo.", "error")
        elif not face_id.MIN_FOTOS <= len(fotos) <= MAX_FOTOS:
            flash(f"Subí entre {face_id.MIN_FOTOS} y {MAX_FOTOS} fotos.", "error")
        else:
            # Las fotos se decodifican en memoria y se descartan: solo quedan los embeddings
            imagenes = [cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) for f in fotos]
            if any(img is None for img in imagenes):
                flash("Uno de los archivos no es una imagen válida (usá JPG o PNG).", "error")
            else:
                utiles, sin_cara = face_id.registrar(nombre, imagenes, current_user.username)
                if utiles >= face_id.MIN_FOTOS:
                    aviso = f" {sin_cara} foto(s) no se usaron porque no se veía la cara." if sin_cara else ""
                    flash(f"'{nombre}' quedó registrada con {utiles} fotos.{aviso}", "success")
                else:
                    flash(f"Solo se encontró la cara en {utiles} de {len(imagenes)} fotos. Hacen falta al menos "
                          f"{face_id.MIN_FOTOS}: usá fotos de frente, con buena luz y sin lentes oscuros.", "error")
        return redirect(url_for("admin.personas"))
    return render_template("admin/personas.html", personas=face_id.listar(),
                           min_fotos=face_id.MIN_FOTOS, max_fotos=MAX_FOTOS, disponible=face_id.disponible())


@admin_bp.route("/personas/borrar", methods=["POST"])
@login_required
@admin_required
def borrar_persona():
    nombre = request.form.get("nombre", "")
    if face_id.borrar(nombre):
        flash(f"Se borraron los datos de reconocimiento de '{nombre}'.", "success")
    else:
        flash("Esa persona no está registrada.", "error")
    return redirect(url_for("admin.personas"))


@admin_bp.route("/personas/consentimiento")
@login_required
@admin_required
def consentimiento():
    return render_template("admin/consentimiento.html")


@admin_bp.route("/personas/contrato")
@login_required
@admin_required
def contrato():
    return render_template("admin/contrato.html")


# ═══════════════════════════════════════
#  CAMERAS MANAGEMENT
# ═══════════════════════════════════════

@admin_bp.route("/monitor")
@roles_required("operator", "admin")
def monitor():
    # Las cámaras activas se monitorean juntas. Las "webcam" toman el video en el navegador;
    # las "usb" e "ip" las analiza el servidor (motor.py) y acá solo se miran.
    activas = [c for c in data_store.load_cameras() if c.get("active")]
    return render_template("admin/monitor.html", cameras=activas)

@admin_bp.route("/cameras", methods=["GET", "POST"])
@login_required
def cameras():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        location = request.form.get("location", "").strip()
        # Registrar una cámara usb/ip hace que el servidor la abra: no alcanza con estar logueado
        if current_user.role not in ("operator", "admin"):
            abort(403)
        cam_type = request.form.get("type", "webcam")
        url = request.form.get("url", "").strip()
        if cam_type == "usb":
            url = url or "0"  # número de dispositivo del servidor: 0 es la primera cámara
        
        if not name:
            flash("El nombre de la cámara es obligatorio.", "error")
        elif cam_type not in ("webcam", "ip", "usb"):
            flash("Tipo de cámara desconocido.", "error")
        elif cam_type == "usb" and not (url.isdecimal() and len(url) <= 2):
            flash("Para una cámara USB indicá el número de dispositivo (0, 1, 2...).", "error")
        elif cam_type == "ip" and not url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            flash("Para una cámara IP indicá la dirección (rtsp://... o http://...).", "error")
        else:
            data_store.create_camera(name, location, cam_type, url)
            motor.sincronizar()
            flash(f"Cámara '{name}' registrada.", "success")
        
        return redirect(url_for("admin.cameras"))
    
    all_cameras = data_store.load_cameras()
    return render_template("admin/cameras.html", cameras=all_cameras)

@admin_bp.route("/cameras/<cam_id>/toggle", methods=["POST"])
@roles_required("operator", "admin")
def toggle_camera(cam_id):
    cam = data_store.toggle_camera(cam_id)
    motor.sincronizar()
    if cam:
        state = "activada" if cam["active"] else "desactivada"
        flash(f"Cámara '{cam['name']}' {state}.", "success")
    return redirect(url_for("admin.cameras"))

@admin_bp.route("/cameras/<cam_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_camera(cam_id):
    data_store.delete_camera(cam_id)
    motor.sincronizar()
    flash("Cámara eliminada.", "success")
    return redirect(url_for("admin.cameras"))


def _zonas_validas(zonas):
    """Hasta 10 polígonos de 3 a 20 puntos [x, y] con coordenadas de 0 a 1."""
    def numero(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1
    return (isinstance(zonas, list) and len(zonas) <= 10
            and all(isinstance(z, list) and 3 <= len(z) <= 20
                    and all(isinstance(p, list) and len(p) == 2 and numero(p[0]) and numero(p[1]) for p in z)
                    for z in zonas))


@admin_bp.route("/api/cameras/<cam_id>/zonas", methods=["POST"])
@roles_required("operator", "admin")
def api_zonas(cam_id):
    """Guarda las zonas de exclusión dibujadas en Monitoreo: {"zonas": [[[x, y], ...], ...]}."""
    data = request.get_json(silent=True)
    zonas = data.get("zonas") if isinstance(data, dict) else None
    if not _zonas_validas(zonas):
        return jsonify({"error": "Zonas inválidas: hasta 10 polígonos de 3 a 20 puntos con coordenadas de 0 a 1"}), 400
    if not data_store.set_zonas(cam_id, zonas):
        return jsonify({"error": "Cámara no encontrada"}), 404
    motor.sincronizar()
    return jsonify({"ok": True, "zonas": zonas})


@admin_bp.route("/api/cameras/<cam_id>/frame.jpg")
@roles_required("operator", "admin")
def api_frame(cam_id):
    """Último frame analizado por el servidor, con los recuadros dibujados."""
    jpg = motor.frame(cam_id)
    if jpg is None:
        abort(404)
    return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@admin_bp.route("/api/cameras/<cam_id>/video")
@roles_required("operator", "admin")
def api_video(cam_id):
    """Video en vivo (MJPEG) de una cámara que analiza el servidor, a unos 20 cuadros por segundo."""
    if motor.frame(cam_id) is None:
        abort(404)

    def cuadros():
        while (jpg := motor.frame(cam_id)) is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)

    return Response(cuadros(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@admin_bp.route("/api/motor")
@roles_required("operator", "admin")
def api_motor():
    return jsonify(motor.estado())


# ═══════════════════════════════════════
#  SYSTEM CONFIG
# ═══════════════════════════════════════

@admin_bp.route("/config", methods=["GET", "POST"])
@login_required
@admin_required
def config():
    available_models = glob.glob("*.keras")
    
    if request.method == "POST":
        action = request.form.get("action")
        
        if action == "reset":
            cfg = data_store.reset_config()
            flash("Configuración restaurada a valores por defecto.", "success")
        else:
            cfg = data_store.load_config()
            try:
                cfg["min_confidence"] = int(request.form.get("min_confidence", 51))
                cfg["confirm_threshold"] = int(request.form.get("confirm_threshold", 3))
                cfg["scan_rate_ms"] = int(request.form.get("scan_rate_ms", 100))
                cfg["active_model"] = request.form.get("active_model", "modelo.keras")
                cfg["detectar_casco"] = request.form.get("detectar_casco") == "on"
                cfg["detectar_chaleco"] = request.form.get("detectar_chaleco") == "on"
                cfg["detectar_caida"] = request.form.get("detectar_caida") == "on"
                cfg["caida_segundos"] = max(3, min(120, int(request.form.get("caida_segundos", 10))))
                if not (cfg["detectar_casco"] or cfg["detectar_chaleco"]):
                    flash("Tiene que quedar activa al menos una detección (casco o chaleco).", "error")
                    return redirect(url_for("admin.config"))
                
                # Clamp values
                cfg["min_confidence"] = max(50, min(99, cfg["min_confidence"]))
                cfg["confirm_threshold"] = max(1, min(20, cfg["confirm_threshold"]))
                cfg["scan_rate_ms"] = max(50, min(1000, cfg["scan_rate_ms"]))
                
                data_store.save_config(cfg)
                flash("Configuración guardada exitosamente.", "success")
            except (ValueError, TypeError):
                flash("Error en los valores ingresados.", "error")
        
        return redirect(url_for("admin.config"))
    
    cfg = data_store.load_config()
    return render_template("admin/config.html", config=cfg, models=available_models)


# ═══════════════════════════════════════
#  AUDIT LOG
# ═══════════════════════════════════════

@admin_bp.route("/audit")
@login_required
def audit():
    event_type = request.args.get("type", "all")
    camera = request.args.get("camera", "all")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    page = int(request.args.get("page", 1))
    
    events, total = data_store.get_filtered_events(
        event_type=event_type,
        camera=camera,
        date_from=date_from,
        date_to=date_to,
        page=page,
        per_page=25
    )
    
    total_pages = max(1, (total + 24) // 25)
    all_cameras = data_store.load_cameras()
    
    return render_template("admin/audit.html",
        events=events,
        total=total,
        page=page,
        total_pages=total_pages,
        filters={
            "type": event_type,
            "camera": camera,
            "date_from": date_from,
            "date_to": date_to
        },
        cameras=all_cameras
    )

@admin_bp.route("/audit/export")
@login_required
def audit_export():
    event_type = request.args.get("type", "all")
    camera = request.args.get("camera", "all")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    
    events, _ = data_store.get_filtered_events(
        event_type=event_type,
        camera=camera,
        date_from=date_from,
        date_to=date_to,
        page=1,
        per_page=99999
    )
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Fecha/Hora", "Tipo", "Confianza (%)", "Cámara", "Ubicación", "Persona"])
    for e in events:
        writer.writerow([
            e.get("timestamp", ""),
            e.get("type", ""),
            e.get("confidence", ""),
            e.get("camera", ""),
            e.get("ubicacion", ""),
            e.get("persona", "")
        ])
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=auditoria_rac.csv"}
    )


# ═══════════════════════════════════════
#  API — Receive events from detector
# ═══════════════════════════════════════

# Tipos que puede mandar el navegador. Zona prohibida y hombre caído solo los decide el servidor (motor.py).
VALID_EVENT_TYPES = ("con_casco", "sin_casco", "sin_chaleco")
# Infracciones: guardan persona y foto, y disparan la alarma
ALERT_TYPES = data_store.ALERT_TYPES
EVENT_LABELS = {"con_casco": "Con Casco", "sin_casco": "Sin Casco", "sin_chaleco": "Sin Chaleco",
                "zona_prohibida": "Zona Prohibida", "hombre_caido": "Hombre Caído"}


@admin_bp.app_context_processor
def _etiquetas_de_evento():
    return {"EVENT_LABELS": EVENT_LABELS}

# Cámara implícita del detector web (index.html la envía siempre).
DEFAULT_CAMERA = "webcam"


def validate_event_payload(data):
    """Devuelve (event_type, confidence, camera, None) o (None, None, None, error)."""
    if not isinstance(data, dict):
        return None, None, None, "Se esperaba un objeto JSON"

    event_type = data.get("type")
    if event_type not in VALID_EVENT_TYPES:
        return None, None, None, f"'type' debe ser uno de {list(VALID_EVENT_TYPES)}"

    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, None, None, "'confidence' debe ser un número"
    confidence = float(confidence)
    if math.isnan(confidence) or not 0 <= confidence <= 100:
        return None, None, None, "'confidence' debe estar entre 0 y 100"

    camera = data.get("camera", DEFAULT_CAMERA)
    if not isinstance(camera, str) or not camera.strip():
        return None, None, None, "'camera' debe ser un texto"
    camera = camera.strip()
    if camera != DEFAULT_CAMERA:
        known = set()
        for c in data_store.load_cameras():
            known.add(c.get("id"))
            known.add(c.get("name"))
        if camera not in known:
            return None, None, None, "'camera' no corresponde a una cámara registrada"

    return event_type, confidence, camera, None


# Las fotos de alertas se guardan en WebP: a igual calidad pesan bastante menos que JPEG
CALIDAD_WEBP = 80


def a_webp(imagen):
    """Bytes WebP de una imagen (bytes JPEG/PNG/WebP o matriz de OpenCV), o None si no se puede.
    Siempre se vuelve a codificar: nunca se guardan bytes tal cual llegan."""
    if imagen is None:
        return None
    if isinstance(imagen, (bytes, bytearray)):
        imagen = cv2.imdecode(np.frombuffer(imagen, np.uint8), cv2.IMREAD_COLOR)
        if imagen is None:
            return None
    ok, webp = cv2.imencode(".webp", imagen, [cv2.IMWRITE_WEBP_QUALITY, CALIDAD_WEBP])
    return webp.tobytes() if ok else None


def foto_de_alerta(data):
    """WebP validado de la captura que manda el detector ("data:image/jpeg;base64,..."),
    o None si no vino o no es una imagen. Se vuelve a codificar: nunca se guardan
    bytes tal cual los manda el cliente."""
    foto = data.get("foto")
    if not isinstance(foto, str) or "," not in foto:
        return None
    try:
        crudo = base64.b64decode(foto.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError):
        return None
    return a_webp(crudo)


def datos_de_camara(camera):
    """(nombre para mostrar, ubicación) de una cámara registrada por id o nombre."""
    for c in data_store.load_cameras():
        if camera in (c.get("id"), c.get("name")):
            return c.get("name", camera), c.get("location", "")
    return ("Detector web" if camera == DEFAULT_CAMERA else camera), ""


def registrar_evento(event_type, confidence, camera, persona=None, foto=None):
    """Guarda el evento y avisa: alarma física y webhook. Lo usan la API (el navegador decide)
    y el motor de cámaras del servidor (motor.py). camera: id o nombre de una cámara registrada.
    Las alertas guardan además quién era, dónde fue y una foto del momento (en WebP).
    foto: bytes de imagen; el motor manda el JPEG del cuadro y acá se pasa a WebP."""
    camera, ubicacion = datos_de_camara(camera)
    extra = {"ubicacion": ubicacion} if ubicacion else {}
    if event_type in ALERT_TYPES:
        extra["persona"] = persona.strip()[:120] if isinstance(persona, str) and persona.strip() else "Sin identificar"
    foto_webp = None
    if foto and event_type in ALERT_TYPES:
        # La API ya la recodificó en foto_de_alerta(); el JPEG del motor se pasa a WebP acá
        foto_webp = foto if foto[:4] == b"RIFF" and foto[8:12] == b"WEBP" else a_webp(foto)
    event = data_store.append_event(event_type, confidence, camera, foto_webp=foto_webp, **extra)

    # Despachar comando al hardware según el tipo de evento
    if event_type in ALERT_TYPES:
        enviar_comando_esp32("R")   # LED Rojo + Buzzer ON
    elif event_type == "con_casco":
        enviar_comando_esp32("G")   # LED Verde + Buzzer OFF

    # Webhook n8n: solo en infracciones. timeout=3s y excepción silenciosa.
    if event_type in ALERT_TYPES and N8N_WEBHOOK_URL:
        n8n_payload = {
            "sensor":        camera,
            "persona":       event.get("persona", "Sin identificar"),
            "alerta":        round(confidence),
            "ubicacion":     ubicacion or camera,
            "mensaje":       (f"Trabajador {event_type.replace('_', ' ')}" if event_type.startswith("sin_")
                              else EVENT_LABELS[event_type]) + f" (confianza: {confidence:.1f}%)",
            "tipo_riesgo":   event_type,
            "confianza_raw": confidence,
            "timestamp":     event.get("timestamp", ""),
        }
        n8n_headers = {}
        if N8N_WEBHOOK_TOKEN:
            n8n_headers["Authorization"] = f"Bearer {N8N_WEBHOOK_TOKEN}"
        # En segundo plano: un n8n lento no frena ni la respuesta ni el análisis de la cámara
        threading.Thread(target=_avisar_n8n, args=(n8n_payload, n8n_headers), daemon=True).start()
    return event


def _avisar_n8n(payload, headers):
    try:
        requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=3)
    except Exception:
        pass  # n8n inaccesible no debe romper el flujo de RAC


motor.registrar_evento = registrar_evento


@admin_bp.route("/alertas/<event_id>.jpg")   # enlaces viejos: sirve la foto que haya (WebP o JPEG)
@admin_bp.route("/alertas/<event_id>.webp")  # va abajo para registrarse primero: url_for arma .webp
@login_required
def foto_alerta(event_id):
    # El id es un uuid generado por el servidor; cualquier otra cosa se rechaza
    if not all(ch in "0123456789abcdef-" for ch in event_id) or len(event_id) != 36:
        abort(404)
    foto = data_store.existing_alert_photo(event_id)
    if not foto:
        abort(404)
    return send_file(foto[0], mimetype=foto[1])


@admin_bp.route("/api/events", methods=["POST"])
@login_required
def api_add_event():
    try:
        data = request.get_json(silent=True)
        event_type, confidence, camera, error = validate_event_payload(data)
        if error:
            return jsonify({"error": error}), 400

        persona = data.get("persona") if event_type in ALERT_TYPES else None
        foto = foto_de_alerta(data) if event_type in ALERT_TYPES else None
        event = registrar_evento(event_type, confidence, camera, persona, foto)

        return jsonify({"ok": True, "event": event})
    except HTTPException:
        raise  # 413, etc.: que Flask responda con el código correcto
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════
#  API — Get config for detector
# ═══════════════════════════════════════

@admin_bp.route("/api/config", methods=["GET"])
@login_required
def api_get_config():
    cfg = data_store.load_config()
    return jsonify(cfg)


# ═══════════════════════════════════════
#  API — Real-time stats for dashboard polling
# ═══════════════════════════════════════

@admin_bp.route("/api/stats", methods=["GET"])
@login_required
def api_get_stats():
    stats = data_store.get_stats()
    return jsonify(stats)


# ═══════════════════════════════════════
#  API — Recent events for live feed
# ═══════════════════════════════════════

@admin_bp.route("/api/recent-events", methods=["GET"])
@login_required
def api_recent_events():
    limit = int(request.args.get("limit", 10))
    events, _ = data_store.get_filtered_events(page=1, per_page=limit)
    return jsonify(events)
