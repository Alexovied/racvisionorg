from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import keras
import base64
import os

app = Flask(__name__)

# Cargar el modelo de Keras
model_path = "modelo.keras"
if not os.path.exists(model_path):
    # Si no existe modelo.keras, buscar el archivo .keras más reciente en el directorio
    import glob
    model_files = glob.glob("*.keras")
    if len(model_files) > 0:
        model_files.sort(key=os.path.getmtime, reverse=True)
        model_path = model_files[0]
        print(f"[+] 'modelo.keras' no encontrado. Cargando el más reciente: {model_path}")
    else:
        print("[!] No se encontraron modelos (.keras). Entrena uno primero ejecutando ea.py.")
        exit(1)

print(f"[+] Cargando modelo: {model_path}")
model = keras.models.load_model(model_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def crop_head(img):
    if img is None:
        return None, None
    h_orig, w_orig = img.shape[:2]
    max_dim = 500  # Resolución optimizada para detección súper rápida
    scale = max_dim / max(h_orig, w_orig)
    if scale < 1.0:
        img_resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
    else:
        img_resized = img
        scale = 1.0

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces_flipped) > 0:
            h_img, w_img = gray.shape
            faces = []
            for (xf, yf, wf, hf) in faces_flipped:
                real_x = w_img - (xf + wf)
                faces.append([real_x, yf, wf, hf])
                
    valid_faces = [f for f in faces if f[1] + f[3]/2 < img_resized.shape[0] * 0.70]
    
    if len(valid_faces) > 0:
        valid_faces = sorted(valid_faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = valid_faces[0]
        
        # Mapear coordenadas de vuelta al tamaño original
        x_orig = int(x / scale)
        y_orig = int(y / scale)
        w_orig_box = int(w / scale)
        h_orig_box = int(h / scale)
        
        y_start = max(0, y_orig - int(h_orig_box * 0.45))
        y_end = min(h_orig, y_orig + h_orig_box + int(h_orig_box * 0.1))
        x_start = max(0, x_orig - int(w_orig_box * 0.1))
        x_end = min(w_orig, x_orig + w_orig_box + int(w_orig_box * 0.1))
        return img[y_start:y_end, x_start:x_end], [x_start, y_start, x_end - x_start, y_end - y_start]
    return None, None

@app.route("/")
def index():
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
            gray_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            has_face = False
            
        gray_test = cv2.resize(gray_test, (64, 64))
        gray_test = np.asarray(gray_test, dtype='float32') / 255.0
        gray_test = gray_test.reshape(1, 64, 64, 1)
        
        # Inferencia rápida
        result = model(gray_test, training=False).numpy()[0]
        con_casco = float(result[0] * 100)
        sin_casco = float(result[1] * 100)
        
        return jsonify({
            "has_face": has_face,
            "con_casco": con_casco,
            "sin_casco": sin_casco,
            "bbox": bbox
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Iniciar la aplicación web en el puerto 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
