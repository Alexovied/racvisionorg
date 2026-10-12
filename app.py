from flask import Flask, render_template, request, jsonify
from flask_login import login_required
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from hardware import init_esp32, enviar_comando_esp32
import cv2
import numpy as np
import base64
import os
import secrets
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
print("[+] Patch GlorotUniform aplicado")
# ────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Límite de tamaño por request: un frame JPEG 640x480 en base64 pesa ~100 KB.
# Evita DoS por imágenes gigantes en /predict (Flask responde 413).
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# SECRET_KEY for sessions (Flask-Login)
# Prioridad: variable de entorno SECRET_KEY (recomendado en Render, cuyo disco es
# efímero) y, si no está, el archivo data/.secret_key (se crea la primera vez).
secret_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", ".secret_key")
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

# Register admin Blueprint & Flask-Login
from admin import admin_bp, init_login, roles_required
app.register_blueprint(admin_bp)
init_login(app)

# Inicializar conexión con ESP32 (falla silenciosamente si no está conectado)
init_esp32()

# Cargar el modelo Keras personalizado
model_path = "modelo.keras"
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


@app.route("/")
def landing():
    return render_template("landing.html")

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({"error": "El request supera el tamaño máximo (2 MB)"}), 413


@app.route("/app")
@roles_required("operator", "admin")
def detector():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@login_required
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
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "No se pudo decodificar el frame"}), 400

        # Detector de cabezas (YOLO): ve el casco desde cualquier ángulo, aunque
        # la persona esté de espaldas o se tape la cara. Se mantienen las claves
        # has_face/faces para que el frontend no cambie.
        if helmet_detector.disponible():
            cabezas = helmet_detector.detectar(img)
            return jsonify({
                "has_face": len(cabezas) > 0,
                "faces": [{"con_casco": c["p_casco"] * 100,
                           "sin_casco": c["p_sin"] * 100,
                           "bbox": c["bbox"]} for c in cabezas]
            })

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
                "faces": response_faces
            })
        else:
            return jsonify({
                "has_face": False,
                "faces": []
            })

    except HTTPException:
        raise  # 413 y otros errores HTTP con su código real
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/reset_buffer", methods=["POST"])
@login_required
def reset_buffer():
    # Ya no hay buffer en el backend, se mantiene por compatibilidad con el frontend
    return jsonify({"ok": True})


# ═══════════════════════════════════════
#  API — Control directo de alarma física
# ═══════════════════════════════════════

@app.route("/api/alarm", methods=["POST"])
@login_required
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
    app.run(host="0.0.0.0", port=port, debug=debug)
