import os
# El detector es público: una imagen de pocos KB puede descomprimirse a varios GB. OpenCV rechaza
# las que pasen de este tamaño (unas 4 veces Full HD). Tiene que ir antes de importar cv2.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(1920 * 1080 * 4))
from flask import Flask, render_template, request, jsonify, Response, session, abort
from flask_login import login_required, current_user
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from hardware import init_esp32, enviar_comando_esp32
import cv2
import numpy as np
import base64
import os
import secrets
import time
import uuid
from collections import deque
import keras
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ── PATCH DE COMPATIBILIDAD ─────────────────────────────────────────────────
# El modelo fue guardado con una versión de Keras 3 que almacena 'input_axes' y
# 'output_axes' en GlorotUniform, pero la versión instalada no los acepta en
# __init__. Los ignoramos de forma segura (no afectan el comportamiento de
# inferencia, solo son metadatos de entrenamiento).
_orig_glorot_init = keras.initializers.GlorotUniform.__init__

def _patched_glorot_init(self, seed=None, **kwargs):
    kwargs.pop('input_axes', None)
    kwargs.pop('output_axes', None)
    _orig_glorot_init(self, seed=seed)

keras.initializers.GlorotUniform.__init__ = _patched_glorot_init

# En el VPS (Keras 3.12) el modelo trae además argumentos de BatchNormalization
# y Dense que esa versión no acepta. También se ignoran.
_orig_bn_init = keras.layers.BatchNormalization.__init__

def _patched_bn_init(self, *args, **kwargs):
    for k in ("renorm", "renorm_clipping", "renorm_momentum"):
        kwargs.pop(k, None)
    _orig_bn_init(self, *args, **kwargs)

keras.layers.BatchNormalization.__init__ = _patched_bn_init

_orig_dense_init = keras.layers.Dense.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _orig_dense_init(self, *args, **kwargs)

keras.layers.Dense.__init__ = _patched_dense_init
print("[+] Patches de compatibilidad Keras aplicados")
# ────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # templates/ y static/ están en la raíz

# Límite de tamaño por request: un frame JPEG 640x480 en base64 pesa ~100 KB.
# Evita DoS por imágenes gigantes en /predict (Flask responde 413).
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# SECRET_KEY for sessions (Flask-Login)
# Prioridad: variable de entorno SECRET_KEY (recomendado en Render, cuyo disco es
# efímero) y, si no está, el archivo data/.secret_key (se crea la primera vez).
secret_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", ".secret_key")
os.makedirs(os.path.dirname(secret_key_path), exist_ok=True)
if os.environ.get("SECRET_KEY"):
    app.secret_key = os.environ["SECRET_KEY"]
elif os.path.exists(secret_key_path):
    with open(secret_key_path, "r") as f:
        app.secret_key = f.read().strip()
else:
    app.secret_key = secrets.token_hex(32)
    with open(secret_key_path, "w") as f:
        f.write(app.secret_key)

# Cookie de sesión: no viaja en POST de otros sitios (Lax) y, con gunicorn
# (producción, detrás de HTTPS), solo por HTTPS. `python src/app.py` es desarrollo local por HTTP.
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = __name__ != "__main__"


# CSRF: todo POST debe traer el token de la sesión, en el campo csrf_token
# (formularios) o en el header X-CSRFToken (fetch). Los tests lo apagan con WTF_CSRF_ENABLED.
def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def verificar_csrf():
    if request.method != "POST" or not app.config.get("WTF_CSRF_ENABLED", True):
        return
    # Los wearables se autentican con su token en el header, sin cookie de sesión: no hay CSRF posible
    if request.path == "/admin/api/dispositivos/telemetria":
        return
    enviado = request.headers.get("X-CSRFToken") or request.form.get("csrf_token", "")
    esperado = session.get("csrf_token")
    if not esperado or not secrets.compare_digest(enviado, esperado):
        abort(403, "Token CSRF inválido o ausente. Recargá la página e intentá de nuevo.")

# Register admin Blueprint & Flask-Login
import admin
from admin import admin_bp, init_login, roles_required
from sgep import sgep_bp
app.register_blueprint(admin_bp)
app.register_blueprint(sgep_bp)
admin.iniciar_reporte_diario()   # reporte por Telegram: solo con REPORTE_HORA y TELEGRAM_* configurados
init_login(app)

# Inicializar conexión con ESP32 (falla silenciosamente si no está conectado)
init_esp32()

# Cargar el modelo Keras personalizado
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelos_app", "modelo.keras")
model = None

if not os.path.exists(model_path):
    print(f"[!] No se encontró {model_path}. El detector no funcionará, pero el panel admin sí.")
else:
    print(f"[+] Cargando modelo: {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
        print("[+] Modelo cargado exitosamente.")
    except Exception as e:
        print(f"[!] Error al cargar el modelo: {e}")
        print("[!] El detector no funcionará, pero el panel admin sí.")

from face_detector import detect_faces
import helmet_detector
import face_id
import epp
import data_store
import motor

# (Suavizado temporal eliminado para manejar múltiples rostros independientemente en el frontend)


# CLAHE compartido para mejorar contraste en condiciones de minería
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def crop_head(img):
    if img is None:
        return []
    h_orig, w_orig = img.shape[:2]

    results = []
    # detect_faces ya devuelve las cajas en coordenadas originales, de mayor a menor
    for x_orig, y_orig, w_orig_box, h_orig_box in detect_faces(img):
        # Expandir MUY agresivo hacia arriba para capturar el casco completo.
        # El detector encuentra la cara (ojos/nariz), el casco está ENCIMA.
        # Con factor 2.0 tomamos el doble de la altura de la cara hacia arriba.
        y_start = max(0, y_orig - int(h_orig_box * 2.0))
        y_end   = min(h_orig, y_orig + h_orig_box + int(h_orig_box * 0.2))
        x_start = max(0, x_orig - int(w_orig_box * 0.3))
        x_end   = min(w_orig, x_orig + w_orig_box + int(w_orig_box * 0.3))
        crop = img[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            continue
        # CLAHE en LAB para mejorar contraste sin afectar color
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = _clahe.apply(l)
        crop_eq = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
        crop_rgb = cv2.cvtColor(crop_eq, cv2.COLOR_BGR2RGB)
        results.append((crop_rgb, [x_start, y_start, x_end - x_start, y_end - y_start]))
    return results


@app.route("/robots.txt")
def robots():
    # Solo la página pública se indexa; el panel y el detector piden login
    texto = ("User-agent: *\nAllow: /$\nAllow: /static/\n"
             "Disallow: /admin\nDisallow: /predict\nDisallow: /api/\n"
             "Sitemap: https://racvision.org/sitemap.xml\n")
    return Response(texto, mimetype="text/plain")

@app.route("/sitemap.xml")
def sitemap():
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
           '<url><loc>https://racvision.org/</loc><changefreq>monthly</changefreq><priority>1.0</priority></url>'
           '</urlset>\n')
    return Response(xml, mimetype="application/xml")

@app.route("/favicon.ico")
def favicon():
    # Los navegadores lo piden siempre en la raíz, aunque la página no lo enlace
    return app.send_static_file("favicon.ico")

@app.route("/")
def landing():
    # Idioma: ?lang=es|en; si no viene, el del navegador (español por defecto)
    lang = request.args.get("lang")
    if lang not in ("es", "en"):
        lang = request.accept_languages.best_match(["es", "en"]) or "es"
    return render_template("landing.html", lang=lang)

@app.route("/privacidad")
def privacidad():
    return render_template("legal/privacidad.html", pagina="privacidad")

@app.route("/cookies")
def cookies():
    return render_template("legal/cookies.html", pagina="cookies")

@app.route("/terminos")
def terminos():
    return render_template("legal/terminos.html", pagina="terminos")

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({"error": "El request supera el tamaño máximo (2 MB)"}), 413


# Caras de prueba del detector: en memoria, por visita y por media hora. Nunca van a disco.
_demo = {}  # id de la visita -> (vence, nombre, embeddings)
DEMO_MINUTOS, DEMO_MAX = 30, 200


def _conocidos():
    """Caras contra las que se identifica en /predict: la de prueba de esta visita y, solo
    con sesión iniciada, las personas dadas de alta (un visitante no ve nombres del personal)."""
    demo = _demo.get(session.get("demo"))
    propias = [(demo[1], demo[2])] if demo and demo[0] > time.time() else []
    anonimo = data_store.load_config().get("modo_anonimo")
    return (face_id._personas() if current_user.is_authenticated and not anonimo else []) + propias


@app.route("/api/demo/cara", methods=["POST", "DELETE"])
def demo_cara():
    """Alta temporal de una cara para probar el reconocimiento: {"nombre", "fotos": [dataURL x3..6]}."""
    if request.method == "DELETE":
        _demo.pop(session.pop("demo", None), None)
        return jsonify({"ok": True})
    data = request.get_json(silent=True)
    fotos = data.get("fotos") if isinstance(data, dict) else None
    nombre = str(data.get("nombre") or "").strip()[:40] if isinstance(data, dict) else ""
    if not nombre or not isinstance(fotos, list) or not 3 <= len(fotos) <= 6:
        return jsonify({"error": "Hace falta un nombre y de 3 a 6 fotos"}), 400
    ahora = time.time()
    for k in [k for k, v in _demo.items() if v[0] <= ahora]:
        _demo.pop(k)
    embs = []
    for f in fotos:  # de a una: nunca hay más de una foto decodificada en memoria
        try:
            crudo = base64.b64decode(f.split(",", 1)[1])
            img = cv2.imdecode(np.frombuffer(crudo, np.uint8), cv2.IMREAD_COLOR) if crudo else None
        except (AttributeError, IndexError, ValueError, cv2.error):
            return jsonify({"error": "Foto inválida"}), 400
        if img is not None:
            embs += face_id._embeddings([img])[0]
    if len(embs) < 3:
        return jsonify({"error": f"Solo se vio la cara en {len(embs)} de {len(fotos)} fotos. "
                                 "Probá de frente y con buena luz."}), 400
    if len(_demo) >= DEMO_MAX and session.get("demo") not in _demo:
        _demo.pop(min(_demo, key=lambda k: _demo[k][0]))  # cupo lleno: sale la prueba más vieja
    token = session.setdefault("demo", uuid.uuid4().hex)
    _demo[token] = (ahora + DEMO_MINUTOS * 60, nombre, np.array(embs, dtype=np.float32))
    return jsonify({"ok": True, "nombre": nombre, "minutos": DEMO_MINUTOS})


@app.route("/app")
def detector():
    # Público, para que cualquiera pueda probarlo. Sin sesión de operador no se guarda nada:
    # ni eventos en auditoría, ni fotos, ni alarma (la página no los manda y la API los rechaza).
    registra = current_user.is_authenticated and current_user.role in ("operator", "admin")
    return render_template("index.html", cfg=data_store.load_config(), registra=registra)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None and not helmet_detector.disponible():
            return jsonify({"error": "Modelo no cargado. Verifique la compatibilidad de versiones de Keras."}), 503

        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("image"), str) or "," not in data["image"]:
            return jsonify({"error": "No se envió la imagen"}), 400

        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        try:
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) if image_bytes else None
        except cv2.error:  # imagen demasiado grande o dañada
            img = None

        if img is None:
            return jsonify({"error": "No se pudo decodificar el frame"}), 400

        # Casco y chaleco son dos detecciones independientes que se prenden desde
        # Configuración. Cada persona trae solo las claves de lo que está activo:
        # con_casco/sin_casco (casco) y chaleco (True / False / None si no se ve el torso).
        cfg = data_store.load_config()
        modos = {"casco": bool(cfg["detectar_casco"]), "chaleco": bool(cfg["detectar_chaleco"])}

        # Detección con YOLO (cabezas y chaleco): la misma que usa el motor de cámaras del servidor
        if helmet_detector.disponible() or (not modos["casco"] and modos["chaleco"] and epp.disponible()):
            # "caida": la página lo pide un par de veces por segundo (es otra pasada de red)
            caida = bool(data.get("caida")) and bool(cfg["detectar_caida"])
            info = {}
            caras, personas = motor.analizar(img, modos, con_personas=caida, conocidos=_conocidos(), info=info)
            respuesta = {"has_face": len(caras) > 0, "modos": modos, "faces": caras,
                         "inclinacion": round(info.get("inclinacion", 0.0), 1)}
            if caida:
                respuesta["acostados"] = motor.acostados(img, personas, caras, modos, info.get("cuerpos"),
                                                         info.get("inclinacion", 0.0))
            return jsonify(respuesta)

        # Respaldo sin detector_casco.onnx: cara (YuNet) + clasificador Keras
        faces_data = crop_head(img)

        if len(faces_data) > 0:
            batch_images = []
            bboxes = []
            
            for cropped, bbox in faces_data:
                rgb_test = cv2.resize(cropped, (128, 128))
                rgb_test = np.asarray(rgb_test, dtype='float32')
                rgb_test = mobilenet_preprocess(rgb_test)
                batch_images.append(rgb_test)
                bboxes.append(bbox)

            batch_images = np.array(batch_images)
            results = model(batch_images, training=False).numpy()
            
            response_faces = []
            for i in range(len(faces_data)):
                raw_con = float(results[i][0] * 100)
                raw_sin = float(results[i][1] * 100)
                
                response_faces.append({
                    "con_casco": raw_con,
                    "sin_casco": raw_sin,
                    "bbox": bboxes[i]
                })

            return jsonify({
                "has_face": True,
                "modos": {"casco": True, "chaleco": False},
                "faces": response_faces
            })
        else:
            return jsonify({
                "has_face": False,
                "modos": {"casco": True, "chaleco": False},
                "faces": []
            })

    except HTTPException:
        raise  # 413 y otros errores HTTP con su código real
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error interno al analizar el frame"}), 500  # el detalle queda en la consola

@app.route("/reset_buffer", methods=["POST"])
def reset_buffer():
    # Ya no hay buffer en el backend, se mantiene por compatibilidad con el frontend
    return jsonify({"ok": True})


# ═══════════════════════════════════════
#  API — Control directo de alarma física
# ═══════════════════════════════════════

@app.route("/api/alarm", methods=["POST"])
@roles_required("operator", "admin")
def api_alarm():
    """
    Endpoint dedicado para activar/desactivar la alarma física del ESP32.
    Cuerpo JSON esperado:
        { "state": "sin_casco" }   -> envía 'R' (LED Rojo + Buzzer)
        { "state": "con_casco" }   -> envía 'G' (LED Verde + Silencio)
        { "state": "off" }         -> envía 'O' (Apagar todo)
    """
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "state" not in data:
            return jsonify({"error": "Campo 'state' requerido"}), 400

        state = data["state"]
        cmd_map = {
            "sin_casco": "R",
            "sin_chaleco": "R",
            "con_casco": "G",
            "off":       "O",
        }
        cmd = cmd_map.get(state)
        if cmd is None:
            return jsonify({"error": f"Estado desconocido: {state}"}), 400

        sent = enviar_comando_esp32(cmd)
        return jsonify({"ok": True, "cmd": cmd, "sent_to_hardware": sent})
    except HTTPException:
        raise
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    # El depurador de Werkzeug permite ejecutar código: solo con FLASK_DEBUG=1.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    # Motor de cámaras del servidor (USB / IP). Con el recargador de debug solo en el proceso
    # que atiende: si no, el proceso padre también abriría las cámaras.
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        motor.iniciar()
    app.run(host="0.0.0.0", port=port, debug=debug)
