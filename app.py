from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow import keras
import base64
import os

app = Flask(__name__)

# Cargar el modelo de Keras
model_path = "modelo_20260707_031839.keras"
if not os.path.exists(model_path):
    print(f"[!] No se encontró {model_path}.")
    exit(1)

print(f"[+] Cargando modelo: {model_path}")
model = keras.models.load_model(model_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def crop_head(img):
    if img is None:
        return None, None
    h_orig, w_orig = img.shape[:2]
    max_dim = 500
    scale = max_dim / max(h_orig, w_orig)
    if scale < 1.0:
        img_resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
    else:
        img_resized = img
        scale = 1.0

    ih, iw = img_resized.shape[:2]
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Mismos parámetros que ea.py (entrenamiento) para que el recorte sea idéntico
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces_flipped) > 0:
            faces = []
            for (xf, yf, wf, hf) in faces_flipped:
                real_x = iw - (xf + wf)
                faces.append([real_x, yf, wf, hf])

    def is_valid_face(f):
        fx, fy, fw, fh = f[0], f[1], f[2], f[3]
        if fy + fh / 2 >= ih * 0.70:
            return False
        # Rechazar si ocupa más del 70% del ancho (UI del navegador en el fondo)
        if fw > iw * 0.70:
            return False
        # Rechazar aspect ratio absurdo
        ratio = fw / fh if fh > 0 else 0
        if ratio < 0.3 or ratio > 3.0:
            return False
        return True

    valid_faces = [f for f in faces if is_valid_face(f)]

    if len(valid_faces) > 0:
        valid_faces = sorted(valid_faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = valid_faces[0]

        x_orig = int(x / scale)
        y_orig = int(y / scale)
        w_orig_box = int(w / scale)
        h_orig_box = int(h / scale)

        # ⚠️ Mismo margen que ea.py para que el recorte sea igual al entrenamiento
        # Margen SUPERIOR grande (1.2x) para capturar siempre el casco por encima de la cara
        y_start = max(0, y_orig - int(h_orig_box * 1.2))
        y_end = min(h_orig, y_orig + h_orig_box + int(h_orig_box * 0.15))
        x_start = max(0, x_orig - int(w_orig_box * 0.15))
        x_end = min(w_orig, x_orig + w_orig_box + int(w_orig_box * 0.15))
        return img[y_start:y_end, x_start:x_end], [x_start, y_start, x_end - x_start, y_end - y_start]
    return None, None


@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/app")
def detector():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No se envió la imagen"}), 400
            
        # Decodificar la imagen base64
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "No se pudo decodificar el frame"}), 400
            
        cropped, bbox = crop_head(img)
        
        if cropped is not None:
            gray_test = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            has_face = True
        else:
            # Igual que ea.py línea 66: usa la imagen completa si no hay cara
            gray_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            has_face = False

        gray_test = cv2.resize(gray_test, (64, 64))
        gray_test = np.asarray(gray_test, dtype='float32') / 255.0
        gray_test = gray_test.reshape(1, 64, 64, 1)
        
        # Inferencia rápida
        result = model(gray_test, training=False).numpy()[0]
        con_casco = float(result[0] * 100)
        sin_casco = float(result[1] * 100)

        # Determinar confianza: si no hay cara, solo mostramos resultado si el modelo
        # es suficientemente seguro (umbral 60%) para evitar falsos positivos
        show_result = has_face or (max(con_casco, sin_casco) > 60.0)

        return jsonify({
            "has_face": has_face,
            "show_result": show_result,
            "con_casco": con_casco,
            "sin_casco": sin_casco,
            "bbox": bbox
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Iniciar la aplicación web en el puerto 5000
    # debug=False para evitar que el modelo se cargue dos veces (reloader de Flask)
    app.run(host="0.0.0.0", port=5000, debug=False)
