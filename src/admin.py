import glob
import csv
import io
import math
import os
import re
import secrets
import threading
import time
import traceback
import requests
from functools import wraps
from urllib.parse import urlsplit
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    jsonify, Response, send_file, abort
)
import base64
from datetime import datetime
import binascii
import cv2
import numpy as np
from werkzeug.exceptions import HTTPException
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
import aprendizaje
import aviso_mqtt
import aviso_telegram
import data_store
import face_detector
import wearables
import face_id
import revision
import riesgo
import salud
import motor
from hardware import enviar_comando_esp32

# ── Integración n8n ──────────────────────────────────────────────────────────
# Configurar en el .env del servidor (o local):
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

    def get_id(self):
        # La sesión queda atada a la contraseña: al cambiarla, las cookies anteriores dejan de servir.
        # Las cookies de antes de este formato (sin "|") no se aceptan: tras el deploy todos, incluidos los
        # televisores del monitor, vuelven a iniciar sesión una vez (aceptarlas revive cookies robadas).
        return f"{self.id}|{_marca(self._data)}"


def _marca(user_dict):
    # Hash del hash: la cookie de sesión va firmada pero no cifrada, no lleva pedazos del hash real
    return wearables._hash(user_dict["password_hash"])[:16]


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
        uid, _, marca = user_id.partition("|")
        u = data_store.get_user_by_id(uid)
        if u and marca == _marca(u):
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

# Fuerza bruta: 5 contraseñas malas para un usuario desde una IP → 5 minutos sin poder probar.
# shortcut: en memoria y por proceso (se pierde al reiniciar); detrás de un proxy sin ProxyFix la IP
# es la del proxy y el bloqueo queda por usuario. Con varios workers o mucho tráfico, pasar a flask-limiter.
LOGIN_MAX_FALLAS, LOGIN_VENTANA = 5, 300
_intentos = {}
_intentos_lock = threading.Lock()


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("admin.dashboard"))
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        clave = (request.remote_addr, username.lower()[:80])   # acotada: un nombre de 2 MB no queda en memoria
        ahora = time.time()
        with _intentos_lock:
            fallas = [t for t in _intentos.get(clave, []) if ahora - t < LOGIN_VENTANA]
            _intentos[clave] = fallas
        if len(fallas) >= LOGIN_MAX_FALLAS:
            flash("Demasiados intentos fallidos. Esperá unos minutos y probá de nuevo.", "error")
            return render_template("admin/login.html"), 429

        user_data = data_store.get_user_by_username(username)
        if user_data and data_store.verify_password(user_data, password):
            _intentos.pop(clave, None)
            user = User(user_data)
            login_user(user)
            next_page = _safe_next(request.args.get("next"))
            return redirect(next_page or url_for("admin.dashboard"))
        else:
            with _intentos_lock:
                if len(_intentos) > 1000:  # nombres inventados no llenan la memoria
                    for k in [k for k, v in _intentos.items() if not v or ahora - v[-1] > LOGIN_VENTANA]:
                        del _intentos[k]
                _intentos.setdefault(clave, []).append(ahora)
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
            if role and role != "admin" and _es_el_unico_admin(user_id):
                flash("Es el único administrador: creá otro antes de quitarle el rol.", "error")
                return redirect(url_for("admin.users"))
            
            success = data_store.update_user(
                user_id,
                username=username or None,
                password=password if password else None,
                role=role or None
            )
            if success:
                if password and user_id == current_user.id:
                    login_user(User(data_store.get_user_by_id(user_id)))   # su propia sesión sigue abierta
                flash("Usuario actualizado.", "success")
            else:
                flash("Error: el nombre de usuario ya está en uso.", "error")
        
        return redirect(url_for("admin.users"))
    
    all_users = data_store.load_users()
    return render_template("admin/users.html", users=all_users)


def _es_el_unico_admin(user_id):
    admins = [u["id"] for u in data_store.load_users() if u["role"] == "admin"]
    return admins == [user_id]

@admin_bp.route("/users/<user_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    if user_id == current_user.id:
        flash("No puedes eliminar tu propio usuario.", "error")
    elif _es_el_unico_admin(user_id):
        flash("Es el único administrador: no se puede eliminar.", "error")
    elif data_store.delete_user(user_id):
        flash("Usuario eliminado.", "success")
    else:
        flash("El usuario no existe.", "error")
    return redirect(url_for("admin.users"))


# ═══════════════════════════════════════
#  PERSONAS (reconocimiento facial)
# ═══════════════════════════════════════

MAX_FOTOS = 10


def _decodificar(crudo):
    """Imagen de OpenCV a partir de bytes, o None si no es una imagen. Con el cuerpo vacío o una
    foto más grande que OPENCV_IO_MAX_IMAGE_PIXELS, imdecode lanza cv2.error en vez de devolver None."""
    try:
        return cv2.imdecode(np.frombuffer(crudo, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None


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
            imagenes = [_decodificar(f.read()) for f in fotos]
            if any(img is None for img in imagenes):
                flash("Uno de los archivos no es una imagen válida (usá JPG o PNG).", "error")
            else:
                try:
                    utiles, sin_cara = face_id.registrar(nombre, imagenes, current_user.username)
                except RuntimeError as e:   # archivo cifrado sin la clave: no se pisa a nadie
                    flash(f"No se pudo registrar: {e}.", "error")
                    return redirect(url_for("admin.personas"))
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
    try:
        borrada = face_id.borrar(nombre)
    except RuntimeError as e:
        flash(f"No se pudo borrar: {e}.", "error")
        return redirect(url_for("admin.personas"))
    if borrada:
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

@admin_bp.app_template_filter("sin_clave")
def _sin_clave(url):
    """URL de una cámara para mostrar en pantalla: sin usuario:clave ni parámetros (?user=..&pwd=..)."""
    return re.sub(r"\?.*", "?***", re.sub(r"//.*@", "//***@", url or "", count=1))


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
        elif cam_type not in ("webcam", "ip", "usb", "relay"):
            flash("Tipo de cámara desconocido.", "error")
        elif cam_type == "usb" and not (url.isdecimal() and len(url) <= 2):
            flash("Para una cámara USB indicá el número de dispositivo (0, 1, 2...).", "error")
        elif cam_type == "ip" and not url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            flash("Para una cámara IP indicá la dirección (rtsp://... o http://...).", "error")
        elif cam_type == "relay":
            # La cámara está en otra red: la manda herramientas/relay_camara.py con este token.
            # Se guarda solo su hash y se muestra una única vez, como el de los wearables.
            token = "rac_" + secrets.token_urlsafe(24)
            cam = data_store.create_camera(name, location, "relay", "", token_hash=wearables._hash(token))
            flash(f"Cámara '{name}' registrada. Para el relay: --id {cam['id']} --token {token} "
                  "(copialo ahora: no se vuelve a mostrar).", "success")
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
    if data_store.delete_camera(cam_id):
        motor.sincronizar()
        flash("Cámara eliminada.", "success")
    else:
        flash("La cámara no existe.", "error")
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
    """Guarda lo dibujado en Monitoreo: {"zonas": [[[x, y], ...], ...], "exentas": [...] (opcional),
    "horario": {"desde": "07:00", "hasta": "19:00"} o null (opcional: horario en que se exige EPP)}."""
    data = request.get_json(silent=True)
    zonas = data.get("zonas") if isinstance(data, dict) else None
    exentas = data.get("exentas") if isinstance(data, dict) else None
    if not _zonas_validas(zonas) or (exentas is not None and not _zonas_validas(exentas)):
        return jsonify({"error": "Zonas inválidas: hasta 10 polígonos de 3 a 20 puntos con coordenadas de 0 a 1"}), 400
    horario = data.get("horario", False)
    hora = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
    if horario not in (False, None) and not (isinstance(horario, dict) and all(
            isinstance(horario.get(k), str) and hora.match(horario[k]) for k in ("desde", "hasta"))):
        return jsonify({"error": "Horario inválido: {desde, hasta} en HH:MM"}), 400
    if horario:
        horario = {"desde": horario["desde"], "hasta": horario["hasta"]}
    cam = data_store.set_zonas(cam_id, zonas, exentas, horario)
    if not cam:
        return jsonify({"error": "Cámara no encontrada"}), 404
    motor.sincronizar()
    return jsonify({"ok": True, "zonas": zonas, "exentas": cam.get("exentas", []), "horario": cam.get("horario")})


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
        anterior = None
        while (jpg := motor.frame(cam_id)) is not None:
            if jpg is not anterior:  # sin cuadro nuevo no se manda nada
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                anterior = jpg
            time.sleep(0.05)

    return Response(cuadros(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@admin_bp.route("/api/cameras/<cam_id>/cuadro", methods=["POST"])
def api_cuadro(cam_id):
    """Cuadro JPEG de una cámara relay, con su token (no con sesión: no pasa por CSRF)."""
    cam = next((c for c in data_store.load_cameras() if c["id"] == cam_id), None)
    token = _token_del_equipo()
    if (not cam or cam.get("type") != "relay" or not token
            or not secrets.compare_digest(cam.get("token_hash", ""), wearables._hash(token))):
        return jsonify({"error": "Cámara o token inválido"}), 401
    if not cam.get("active"):
        return jsonify({"error": "Cámara desactivada"}), 409
    img = _decodificar(request.get_data())
    if img is None:
        return jsonify({"error": "El cuerpo tiene que ser un JPEG"}), 400
    numero = request.headers.get("X-Cuadro", "")
    if not motor.recibir(cam_id, img, int(numero) if numero.isdecimal() and len(numero) < 12 else None):
        return jsonify({"error": "Cámara desactivada"}), 409
    return jsonify({"ok": True})


@admin_bp.route("/api/motor")
@roles_required("operator", "admin")
def api_motor():
    # _nodo: CPU / RAM / temperatura del equipo (un nodo recalentado o saturado deja de analizar a tiempo)
    return jsonify({**motor.estado(), "_nodo": salud.nodo()})


# ═══════════════════════════════════════
#  SYSTEM CONFIG
# ═══════════════════════════════════════

@admin_bp.route("/config", methods=["GET", "POST"])
@login_required
@admin_required
def config():
    available_models = [os.path.basename(m) for m in glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelos_app", "*.keras"))]
    
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
                cfg["modo_anonimo"] = request.form.get("modo_anonimo") == "on"
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
    page = max(1, request.args.get("page", 1, type=int))
    
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
    todos = data_store.load_events()
    
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
        cameras=all_cameras,
        precision=aprendizaje.precision(todos),
        sugerencia=aprendizaje.sugerir_confianza(todos, data_store.load_config()["min_confidence"]),
        ALERT_TYPES=ALERT_TYPES,
    )

def _celda_csv(valor):
    """Excel ejecuta como fórmula una celda que empieza con = + - @ (o tab / CR): se antepone '."""
    if isinstance(valor, str) and valor.startswith(("=", "+", "-", "@", chr(9), chr(13))):
        return "'" + valor
    return valor


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
        writer.writerow([_celda_csv(e.get(k, "")) for k in
                         ("timestamp", "type", "confidence", "camera", "ubicacion", "persona")])
    
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
                "zona_prohibida": "Zona Prohibida", "hombre_caido": "Hombre Caído",
                "boton_sos": "Botón de Pánico", "hombre_muerto": "Hombre Muerto", "posible_desmayo": "Posible Desmayo",
                "ritmo_cardiaco": "Ritmo Cardíaco", "wearable_sin_senal": "Wearable sin Señal",
                "camara_sin_senal": "Cámara sin señal", "camara_tapada": "Cámara tapada",
                "camara_desenfocada": "Cámara desenfocada"}
SALUD_TYPES = data_store.SALUD_TYPES


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


def anonimizar(foto):
    """JPEG de la foto con las caras pixeladas (modo anónimo), o None si no es una imagen o si no
    se encontró ninguna cara: ante la duda no se guarda la foto (una cara de perfil o tapada que el
    detector no ve quedaría nítida). Se agranda la caja de cada cara para tapar también pelo y orejas."""
    img = _decodificar(foto)
    if img is None:
        return None
    H, W = img.shape[:2]
    caras = face_detector.detect_faces(img)
    if not caras:
        return None
    for x, y, w, h in caras:
        x0, y0 = max(0, int(x - 0.3 * w)), max(0, int(y - 0.4 * h))
        x1, y1 = min(W, int(x + 1.3 * w)), min(H, int(y + 1.3 * h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        chica = cv2.resize(img[y0:y1, x0:x1], (8, 8), interpolation=cv2.INTER_LINEAR)
        img[y0:y1, x0:x1] = cv2.resize(chica, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpg.tobytes() if ok else None


def a_webp(imagen):
    """Bytes WebP de una imagen (bytes JPEG/PNG/WebP o matriz de OpenCV), o None si no se puede.
    Siempre se vuelve a codificar: nunca se guardan bytes tal cual llegan."""
    if imagen is None:
        return None
    if isinstance(imagen, (bytes, bytearray)):
        imagen = _decodificar(imagen)
        if imagen is None:
            return None
    ok, webp = cv2.imencode(".webp", imagen, [cv2.IMWRITE_WEBP_QUALITY, CALIDAD_WEBP])
    return webp.tobytes() if ok else None


def _base64_de(valor):
    """Bytes de un "data:image/jpeg;base64,...", o None si no vino o no es base64."""
    if not isinstance(valor, str) or "," not in valor:
        return None
    try:
        return base64.b64decode(valor.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError):
        return None


def foto_de_alerta(data):
    """WebP validado de la captura que manda el detector ("data:image/jpeg;base64,..."),
    o None si no vino o no es una imagen. Se vuelve a codificar: nunca se guardan
    bytes tal cual los manda el cliente."""
    crudo = _base64_de(data.get("foto"))
    return a_webp(crudo) if crudo else None


def guardar_muestra(evento, foto):
    """Guarda la foto limpia (sin recuadros) de una alerta de cámara para revisarla en
    /admin/revision. En modo anónimo no: ahí no se guarda ninguna cara sin pixelar."""
    if not foto or evento.get("type") not in revision.TIPOS or data_store.load_config().get("modo_anonimo"):
        return None
    return revision.guardar(foto, evento.get("camera"), "alerta", evento["type"], evento["id"])


def datos_de_camara(camera):
    """(nombre para mostrar, ubicación) de una cámara registrada por id o nombre."""
    for c in data_store.load_cameras():
        if camera in (c.get("id"), c.get("name")):
            return c.get("name", camera), c.get("location", "")
    return ("Detector web" if camera == DEFAULT_CAMERA else camera), ""


def _posicion(pos):
    """[x, y] de 0 a 1 (dónde estaba la persona en la imagen) o None si no vino bien."""
    if (isinstance(pos, list) and len(pos) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1 for v in pos)):
        return [round(float(pos[0]), 3), round(float(pos[1]), 3)]
    return None


def registrar_evento(event_type, confidence, camera, persona=None, foto=None, pos=None, extra_campos=None):
    """Guarda el evento y avisa: alarma física y webhook. Lo usan la API (el navegador decide)
    y el motor de cámaras del servidor (motor.py). camera: id o nombre de una cámara registrada.
    Las alertas guardan además quién era, dónde fue y una foto del momento (en WebP).
    foto: bytes de imagen; el motor manda el JPEG del cuadro y acá se pasa a WebP.
    extra_campos: datos propios del origen (los wearables mandan dispositivo, detalle y medición)."""
    camera, ubicacion = datos_de_camara(camera)
    extra = {"ubicacion": ubicacion} if ubicacion else {}
    extra.update(extra_campos or {})
    ubicacion = extra.get("ubicacion", "")
    if data_store.load_config().get("modo_anonimo"):   # nadie con nombre y caras pixeladas en la foto
        persona = None
        foto = anonimizar(foto) if foto else None
    if event_type in ALERT_TYPES:
        extra["persona"] = persona.strip()[:120] if isinstance(persona, str) and persona.strip() else "Sin identificar"
        if _posicion(pos):
            extra["pos"] = _posicion(pos)
    if event_type == "hombre_caido":
        # Si el wearable de alguien avisó caída, desmayo o SOS hace poco, la cámara lo confirma
        quien = extra.get("persona") if extra.get("persona") != "Sin identificar" else None
        previo = wearables.buscar_confirmacion(data_store.load_events(), wearables.TIPOS_DE_CAIDA,
                                               datetime.now(), persona=quien)
        if previo:
            extra["confirmado_por"] = f"wearable {previo['camera']} ({previo.get('persona', '')})"
            extra["evento_relacionado"] = previo["id"]
    foto_webp = None
    if foto and (event_type in ALERT_TYPES or event_type in SALUD_TYPES):
        # La API ya la recodificó en foto_de_alerta(); el JPEG del motor se pasa a WebP acá
        foto_webp = foto if foto[:4] == b"RIFF" and foto[8:12] == b"WEBP" else a_webp(foto)
    event = data_store.append_event(event_type, confidence, camera, foto_webp=foto_webp, **extra)

    # Ya quedó guardado: si un aviso falla (p. ej. no se puede abrir otro hilo) no se propaga, así
    # quien llama solo ve un error cuando no se guardó y reintentar no duplica el evento.
    try:
        # Despachar comando al hardware según el tipo de evento
        if event_type in ALERT_TYPES and event_type != "wearable_sin_senal":
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
        if event_type in ALERT_TYPES or event_type in SALUD_TYPES:
            aviso_telegram.avisar(event, foto_webp)   # sin TELEGRAM_BOT_TOKEN no hace nada
        aviso_mqtt.publicar(event)   # todos los eventos, también "con_casco": sin MQTT_HOST no hace nada
    except Exception:
        traceback.print_exc()
    return event


def _avisar_n8n(payload, headers):
    # n8n caído no debe romper el flujo de RAC, pero queda en el log (si no, nadie se entera)
    try:
        r = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=3)
        if r.status_code >= 400:
            print(f"[!] n8n respondió {r.status_code} a la alerta {payload['tipo_riesgo']}", flush=True)
    except Exception as e:
        # solo el tipo de error: el texto trae la URL del webhook, que es el secreto cuando no hay token
        print(f"[!] n8n no recibió la alerta {payload['tipo_riesgo']}: {type(e).__name__}", flush=True)


motor.registrar_evento = registrar_evento
motor.guardar_muestra = guardar_muestra


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
@roles_required("operator", "admin")   # viewer es solo lectura
def api_add_event():
    try:
        data = request.get_json(silent=True)
        event_type, confidence, camera, error = validate_event_payload(data)
        if error:
            return jsonify({"error": error}), 400

        persona = data.get("persona") if event_type in ALERT_TYPES else None
        foto = foto_de_alerta(data) if event_type in ALERT_TYPES else None
        event = registrar_evento(event_type, confidence, camera, persona, foto, _posicion(data.get("pos")))
        if event_type in revision.TIPOS:
            try:   # la foto sin recuadros, para revisar la alerta; si falla, la alerta ya quedó
                guardar_muestra(event, _base64_de(data.get("foto_limpia")))
            except Exception as e:
                print(f"[!] No se guardó la foto para revisar: {e}", flush=True)

        return jsonify({"ok": True, "event": event})
    except HTTPException:
        raise  # 413, etc.: que Flask responda con el código correcto
    except Exception:
        traceback.print_exc()   # si no, una alerta de webcam que no se pudo guardar se pierde sin rastro
        return jsonify({"error": "Error interno al registrar el evento"}), 500


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
    limit = max(1, min(100, request.args.get("limit", 10, type=int)))
    events, _ = data_store.get_filtered_events(page=1, per_page=limit)
    return jsonify(events)


# ═══════════════════════════════════════
#  WEARABLES: pulso, caídas, botón SOS y hombre muerto (wearables.py)
# ═══════════════════════════════════════

# Lo que salta en el aviso rojo de todas las páginas del panel hasta que alguien lo marque
URGENTES = ("boton_sos", "hombre_muerto", "posible_desmayo", "hombre_caido", "ritmo_cardiaco")
URGENTE_MINUTOS = 15


def alerta_de_wearable(disp, tipo, detalle, lectura):
    """Registra una alerta de un wearable como cualquier otra (panel, alarma, n8n y Telegram).
    Si es una caída y la cámara de su zona vio a alguien en el piso hace poco, va confirmada."""
    extra = {"dispositivo": disp["id"], "detalle": detalle,
             "medicion": {k: lectura[k] for k in ("bpm", "spo2", "actividad", "bateria") if k in lectura}}
    if disp.get("camara"):
        _, ubicacion = datos_de_camara(disp["camara"])
        if ubicacion:
            extra["ubicacion"] = ubicacion
    confianza = wearables.CONFIANZA[tipo]
    if tipo in wearables.TIPOS_DE_CAIDA:
        camara = datos_de_camara(disp["camara"])[0] if disp.get("camara") else None
        vista = wearables.buscar_confirmacion(data_store.load_events(), ("hombre_caido",), datetime.now(),
                                              camara=camara, persona=disp.get("persona"))
        if vista:
            extra["confirmado_por"] = f"cámara {vista['camera']}"
            extra["evento_relacionado"] = vista["id"]
            confianza = max(confianza, 98)
    return registrar_evento(tipo, confianza, disp["nombre"], persona=disp.get("persona"), extra_campos=extra)


def revisar_wearables(ahora=None):
    """Avisa de los equipos que dejaron de mandar datos. La llama el hilo de vigilancia."""
    ahora = time.monotonic() if ahora is None else ahora
    with wearables._lock:
        ids = wearables.callados(wearables._estados, ahora)
    avisados = []
    for disp_id in ids:
        disp = wearables.obtener(disp_id)
        if disp and disp.get("activo", True):
            try:
                avisados.append(alerta_de_wearable(disp, "wearable_sin_senal",
                                                   f"sin datos hace más de {wearables.SIN_SENAL_S} s", {}))
            except Exception:
                # No se pudo guardar (disco, archivo tomado): se reintenta en la próxima vuelta
                # y los demás equipos callados avisan igual.
                traceback.print_exc()
                wearables.estado_de(disp_id).sin_senal = False
    return avisados


_vigilancia = None


def _iniciar_vigilancia():
    """Un hilo que cada 15 s revisa qué wearables se callaron. Arranca con la primera lectura."""
    global _vigilancia
    if _vigilancia is not None:
        return

    def bucle():
        while True:
            time.sleep(15)
            try:
                revisar_wearables()
            except Exception as e:
                print(f"[!] Vigilancia de wearables: {e}", flush=True)

    _vigilancia = threading.Thread(target=bucle, daemon=True, name="vigilancia-wearables")
    _vigilancia.start()


def _token_del_equipo():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Device-Token", "").strip()


@admin_bp.route("/api/dispositivos/telemetria", methods=["POST"])
def api_telemetria():
    """La llama el wearable cada pocos segundos, con su token (no con sesión: no pasa por CSRF).
    Responde qué alertas nuevas generó, para que el equipo vibre y el trabajador sepa que llegó."""
    disp = wearables.por_token(_token_del_equipo())
    if not disp:
        return jsonify({"error": "Token de dispositivo inválido"}), 401
    lectura, error = wearables.validar_lectura(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400
    with wearables._lock:
        nuevas = wearables.evaluar(wearables.estado_de(disp["id"]), lectura, time.monotonic(),
                                   wearables.umbrales_de(disp))
    eventos = []
    for i, (tipo, detalle) in enumerate(nuevas):
        try:
            eventos.append(alerta_de_wearable(disp, tipo, detalle, lectura))
        except Exception:
            # No se pudo guardar (disco, archivo tomado): las que faltan dejan de figurar como ya
            # avisadas, así la próxima lectura (el equipo reintenta con el 500) las vuelve a generar.
            with wearables._lock:
                wearables.estado_de(disp["id"]).activas.difference_update(t for t, _ in nuevas[i:])
            raise
    _iniciar_vigilancia()
    return jsonify({"ok": True, "alertas": [e["type"] for e in eventos], "aviso_recibido": bool(eventos)})


@admin_bp.route("/wearables", methods=["GET", "POST"])
@login_required
def wearables_pagina():
    """Lista de equipos en vivo. El admin registra equipos: el token se muestra una sola vez."""
    if current_user.role in ("pm", "auditor"):
        return redirect(url_for("sgep.dashboard"))
    token_nuevo = None
    if request.method == "POST":
        if current_user.role != "admin":
            flash("Solo un administrador puede registrar equipos.", "error")
            return redirect(url_for("admin.wearables_pagina"))
        nombre = request.form.get("nombre", "").strip()
        persona = request.form.get("persona", "").strip()
        if not nombre or not persona:
            flash("Poné un nombre para el equipo y a quién se lo das.", "error")
        else:
            disp, token = wearables.crear(nombre, persona, request.form.get("camara", ""))
            token_nuevo = {"nombre": disp["nombre"], "token": token}
    return render_template("admin/wearables.html", cameras=data_store.load_cameras(), token_nuevo=token_nuevo,
                           umbrales=wearables.UMBRALES, sin_senal=wearables.SIN_SENAL_S)


@admin_bp.route("/wearables/<disp_id>/token", methods=["POST"])
@login_required
@admin_required
def wearable_token(disp_id):
    token = wearables.regenerar_token(disp_id)
    if not token:
        abort(404)
    disp = wearables.obtener(disp_id)
    return render_template("admin/wearables.html", cameras=data_store.load_cameras(),
                           token_nuevo={"nombre": disp["nombre"], "token": token},
                           umbrales=wearables.UMBRALES, sin_senal=wearables.SIN_SENAL_S)


@admin_bp.route("/wearables/<disp_id>/borrar", methods=["POST"])
@login_required
@admin_required
def wearable_borrar(disp_id):
    if wearables.borrar(disp_id):
        flash("Equipo borrado. Si lo volvés a usar, registralo de nuevo y cargale el token nuevo.", "success")
    else:
        flash("Ese equipo ya no existe.", "error")
    return redirect(url_for("admin.wearables_pagina"))


@admin_bp.route("/wearables/<disp_id>/restablecer", methods=["POST"])
@login_required
@admin_required
def wearable_restablecer(disp_id):
    if wearables.restablecer_umbrales(disp_id):
        flash("Umbrales de fábrica restablecidos para ese equipo.", "success")
    return redirect(url_for("admin.wearables_pagina"))


@admin_bp.route("/api/wearables")
@roles_required("admin", "operator", "viewer")
def api_wearables():
    ahora = time.monotonic()
    return jsonify([wearables.resumen(d, ahora) for d in wearables.cargar()])


@admin_bp.route("/api/alertas-urgentes")
@roles_required("admin", "operator", "viewer")
def api_alertas_urgentes():
    """Alertas graves de los últimos minutos que nadie marcó todavía (el aviso rojo del panel)."""
    desde = datetime.now().timestamp() - URGENTE_MINUTOS * 60
    urgentes = []
    for e in data_store.load_events():
        try:
            if datetime.fromisoformat(e["timestamp"]).timestamp() < desde:
                break
        except (KeyError, ValueError):
            continue
        if e.get("type") in URGENTES and not e.get("feedback"):
            urgentes.append({k: e.get(k) for k in ("id", "type", "camera", "persona", "ubicacion", "detalle",
                                                    "confirmado_por", "timestamp")}
                            | {"etiqueta": EVENT_LABELS.get(e["type"], e["type"])})
    return jsonify(urgentes[:10])


_feedback_lock = threading.Lock()   # shortcut: un solo proceso (gunicorn --workers 1); con más workers haría falta un candado de archivo


@admin_bp.route("/api/events/<event_id>/feedback", methods=["POST"])
@roles_required("admin", "operator")
def api_feedback(event_id):
    """Fue real o falsa alarma: atiende el aviso y le enseña al sistema."""
    data = request.get_json(silent=True) or {}
    veredicto = data.get("veredicto")
    if veredicto not in ("real", "falsa"):
        return jsonify({"error": "'veredicto' debe ser 'real' o 'falsa'"}), 400
    with _feedback_lock:   # leer el veredicto anterior y guardar el nuevo sin que otro pedido se meta en medio
        evento = data_store.get_event(event_id)
        if not evento or evento.get("type") not in ALERT_TYPES:
            return jsonify({"error": "No existe esa alerta"}), 404
        ya_falsa = (evento.get("feedback") or {}).get("veredicto") == "falsa"   # doble clic o dos operadores
        evento = data_store.set_feedback(event_id, veredicto, current_user.username)
    aprendido = wearables.aprender_de_falsa(evento) if veredicto == "falsa" and not ya_falsa else None
    return jsonify({"ok": True, "aprendido": aprendido})


# ═══════════════════════════════════════
#  REVISIÓN DE ALERTAS PARA REENTRENAR (revision.py)
# ═══════════════════════════════════════

@admin_bp.route("/revision")
@roles_required("admin", "operator")
def revision_pagina():
    return render_template("admin/revision.html", resumen=revision.resumen(),
                           anonimo=bool(data_store.load_config().get("modo_anonimo")),
                           etiquetas=EVENT_LABELS)


@admin_bp.route("/api/revision")
@roles_required("admin", "operator")
def api_revision_lista():
    """Fotos para revisar: estado = pendientes (por defecto), revisadas o todas."""
    estado = request.args.get("estado", "pendientes")
    muestras = revision.cargar()
    if estado == "pendientes":
        muestras = [m for m in muestras if m["cajas"] is None]
    elif estado == "revisadas":
        muestras = [m for m in muestras if m["cajas"] is not None]
    campos = ("id", "origen", "tipo", "camera", "timestamp", "veredicto", "event_id")
    return jsonify({"muestras": [{k: m.get(k) for k in campos} | {"revisada": m["cajas"] is not None}
                                 for m in muestras[:500]],
                    "resumen": revision.resumen()})


@admin_bp.route("/api/revision/<muestra_id>")
@roles_required("admin", "operator")
def api_revision_muestra(muestra_id):
    """Una foto con las cajas: las corregidas si ya se revisó, si no las que propone el modelo."""
    m = revision.obtener(muestra_id)
    if not m:
        return jsonify({"error": "No existe esa foto"}), 404
    detecciones = revision.detectar(m)
    evento = data_store.get_event(m["event_id"]) if m.get("event_id") else None
    return jsonify(m | {"detecciones": detecciones,
                        "persona": evento.get("persona") if evento else None,
                        "confianza": evento.get("confidence") if evento else None,
                        "etiqueta": EVENT_LABELS.get(m.get("tipo"), m.get("tipo"))})


@admin_bp.route("/revision/<muestra_id>.webp")
@roles_required("admin", "operator")
def revision_foto(muestra_id):
    if not revision.obtener(muestra_id):
        abort(404)
    return send_file(revision.ruta_foto(muestra_id), mimetype="image/webp")


@admin_bp.route("/api/revision/<muestra_id>", methods=["POST"])
@roles_required("admin", "operator")
def api_revision_guardar(muestra_id):
    m = revision.obtener(muestra_id)
    if not m:
        return jsonify({"error": "No existe esa foto"}), 404
    data = request.get_json(silent=True) or {}
    cajas = revision.validar_cajas(data.get("cajas"))
    if cajas is None:
        return jsonify({"error": "Cajas inválidas"}), 400
    veredicto = data.get("veredicto")
    if m.get("origen") == "alerta":
        if veredicto not in ("real", "falsa"):
            return jsonify({"error": "'veredicto' debe ser 'real' o 'falsa'"}), 400
    else:
        veredicto = None
    revision.revisar(muestra_id, cajas, veredicto, current_user.username)
    return jsonify({"ok": True, "resumen": revision.resumen()})


@admin_bp.route("/api/revision/<muestra_id>/reabrir", methods=["POST"])
@roles_required("admin", "operator")
def api_revision_reabrir(muestra_id):
    if not revision.obtener(muestra_id):
        return jsonify({"error": "No existe esa foto"}), 404
    revision.reabrir(muestra_id)
    return jsonify({"ok": True, "resumen": revision.resumen()})


@admin_bp.route("/api/revision/<muestra_id>/borrar", methods=["POST"])
@roles_required("admin", "operator")
def api_revision_borrar(muestra_id):
    if not revision.obtener(muestra_id) or not revision.borrar(muestra_id):
        return jsonify({"error": "No existe esa foto"}), 404
    return jsonify({"ok": True, "resumen": revision.resumen()})


@admin_bp.route("/api/revision/capturar", methods=["POST"])
@roles_required("admin", "operator")
def api_revision_capturar():
    """Guarda el cuadro actual de una cámara para revisarlo: sirve para los errores que no
    generan alerta (una cabeza sin casco que el modelo vio "con casco"). Webcam: el navegador
    manda la foto; cámara USB / IP: se toma el último cuadro del motor."""
    if data_store.load_config().get("modo_anonimo"):
        return jsonify({"error": "En modo anónimo no se guardan fotos para revisar"}), 409
    data = request.get_json(silent=True) or {}
    cam_id = data.get("camera")
    cam = next((c for c in data_store.load_cameras() if c.get("id") == cam_id), None)
    if not cam:
        return jsonify({"error": "No existe esa cámara"}), 404
    foto = _base64_de(data.get("foto")) if cam.get("type") == "webcam" else motor.cuadro_limpio(cam_id)
    if not foto:
        return jsonify({"error": "No hay imagen de esa cámara"}), 409
    m = revision.guardar(foto, cam.get("name", cam_id), "captura")
    if not m:
        return jsonify({"error": "La foto no es una imagen válida"}), 400
    return jsonify({"ok": True, "id": m["id"]})


@admin_bp.route("/revision/dataset.zip")
@admin_required
def revision_dataset():
    """Dataset YOLO con las fotos revisadas, para reentrenar en Colab."""
    contenido, n = revision.exportar_zip()
    if not n:
        flash("Todavía no hay fotos revisadas para exportar.", "error")
        return redirect(url_for("admin.revision_pagina"))
    nombre = f"rac_revisadas_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    return send_file(io.BytesIO(contenido), mimetype="application/zip", as_attachment=True, download_name=nombre)


# ═══════════════════════════════════════
#  MAPA DE RIESGO Y REPORTE DIARIO (riesgo.py)
# ═══════════════════════════════════════

@admin_bp.route("/riesgo")
@login_required
def riesgo_pagina():
    camaras = [c for c in data_store.load_cameras()]
    return render_template("admin/riesgo.html", cameras=camaras, telegram=aviso_telegram.configurado())


@admin_bp.route("/api/riesgo")
@login_required
def api_riesgo():
    """Matriz día × hora, pronóstico de mañana, tendencia y puntos de una cámara (por nombre)."""
    tipo = request.args.get("tipo")
    tipo = tipo if tipo in ALERT_TYPES else None
    dias = max(1, min(365, request.args.get("dias", 28, type=int)))
    camara = request.args.get("camara") or None
    eventos = data_store.load_events()
    return jsonify({
        "matriz": riesgo.matriz(eventos, dias=dias, tipo=tipo, camara=camara),
        "pronostico": riesgo.pronostico(eventos),
        "tendencia": riesgo.tendencia(eventos),
        "puntos": riesgo.puntos(eventos, camara, dias=dias, tipo=tipo) if camara else [],
        "kpis": riesgo.kpis(eventos, dias=dias),
    })


@admin_bp.route("/riesgo/kpis.csv")
@login_required
def riesgo_kpis_csv():
    """Indicadores por turno y alertas por día, para adjuntar a una auditoría ISO 45001."""
    dias = max(1, min(365, request.args.get("dias", 30, type=int)))
    k = riesgo.kpis(data_store.load_events(), dias=dias)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([f"RAC Vision - indicadores de los últimos {dias} días", f"días sin alertas: {k['dias_sin_alertas']}"])
    w.writerow(["Turno", "Alertas", *[EVENT_LABELS[t] for t in ALERT_TYPES], "Corregidas", "Corrección (min, mediana)"])
    for f in k["turnos"]:
        w.writerow([f["turno"], f["alertas"], *[f["por_tipo"].get(t, 0) for t in ALERT_TYPES], f["corregidas"],
                    "" if f["correccion_min"] is None else f["correccion_min"]])
    w.writerow([])
    w.writerow(["Fecha", "Alertas"])
    for d in k["por_dia"]:
        w.writerow([d["fecha"], d["alertas"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=rac_kpis_{dias}d.csv"})


@admin_bp.route("/reporte")
@login_required
def reporte_pagina():
    try:
        dia = datetime.strptime(request.args.get("fecha", ""), "%Y-%m-%d").date()
    except ValueError:
        dia = None
    r = riesgo.reporte(data_store.load_events(), dia)
    # La IA solo se usa a pedido (?ia=1): cada recarga de la página no gasta una llamada
    usar_ia = request.args.get("ia") == "1" and current_user.role == "admin"   # cada llamada cuesta
    texto = riesgo.redactar(r) if usar_ia else riesgo.texto(r)
    return render_template("admin/reporte.html", r=r, texto=texto, telegram=aviso_telegram.configurado(),
                           ia=riesgo.hay_ia())


@admin_bp.route("/reporte/enviar", methods=["POST"])
@login_required
@admin_required
def reporte_enviar():
    texto = request.form.get("texto", "").strip()
    if not texto:
        flash("El reporte está vacío.", "error")
    elif aviso_telegram.mandar_texto(texto):
        flash("Reporte enviado por Telegram.", "success")
    else:
        flash("No se pudo enviar: revisá TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID.", "error")
    return redirect(url_for("admin.reporte_pagina", fecha=request.form.get("fecha", "")))


_textos_reporte = {}   # fecha → texto ya redactado: si Telegram falla, el reintento no vuelve a pagar la IA


def enviar_reporte_del_dia(dia=None):
    """Arma el reporte de `dia` (ayer por defecto), lo redacta y lo manda por Telegram una sola vez.
    Un .lock (O_EXCL) evita que dos procesos lo manden a la vez y el .enviado se escribe recién
    cuando Telegram lo aceptó: si el proceso muere a mitad, se reintenta en la próxima vuelta."""
    r = riesgo.reporte(data_store.load_events(), dia)
    os.makedirs(data_store.DATA_DIR, exist_ok=True)
    marca = os.path.join(data_store.DATA_DIR, f"reporte-{r['fecha']}.enviado")
    lock = marca + ".lock"
    if os.path.exists(marca):
        return False
    try:
        if time.time() - os.path.getmtime(lock) > 600:   # lock de un proceso que murió
            os.remove(lock)
    except OSError:
        pass
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    try:
        if r["fecha"] not in _textos_reporte:
            _textos_reporte[r["fecha"]] = riesgo.redactar(r)
        if not aviso_telegram.mandar_texto(_textos_reporte[r["fecha"]]):
            return False
        open(marca, "w").close()
        _textos_reporte.pop(r["fecha"], None)
        return True
    finally:
        os.remove(lock)


def iniciar_reporte_diario():
    """Con REPORTE_HORA=8 (y Telegram configurado) manda el reporte de ayer todos los días a esa hora.
    shortcut: arranca un hilo por proceso; con varios workers el .lock y el .enviado en data/ evitan
    duplicados mientras compartan la carpeta. Con varios contenedores, pasarlo a un cron aparte."""
    hora = os.environ.get("REPORTE_HORA", "")
    if not (hora.isdecimal() and 0 <= int(hora) <= 23) or not aviso_telegram.configurado():
        return None

    def bucle():
        while True:
            if datetime.now().hour >= int(hora):
                try:
                    enviar_reporte_del_dia()
                except Exception as e:
                    print(f"[!] Reporte diario: {aviso_telegram.sin_token(e)}", flush=True)
            time.sleep(300)

    hilo = threading.Thread(target=bucle, daemon=True, name="reporte-diario")
    hilo.start()
    return hilo
