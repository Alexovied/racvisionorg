"""
Compara qué ve cada versión del detector.
Corre esto mientras app.py está corriendo, abre la cam web un momento,
y este script captura un frame de la cámara directamente con OpenCV
y lo procesa igual que lo haría camara.py.
Luego lee el último debug_cropped_color.jpg que guardó app.py
y los muestra lado a lado.
"""
import cv2
import numpy as np
from tensorflow import keras
import os

MODEL_PATH = "modelo_20260624_142554.keras"
print(f"Cargando {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def crop_head(img):
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    valid_faces = [f for f in faces if f[1] + f[3]/2 < img.shape[0] * 0.70]
    if len(valid_faces) > 0:
        valid_faces = sorted(valid_faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = valid_faces[0]
        y_start = max(0, y - int(h * 0.45))
        y_end = min(img.shape[0], y + h + int(h * 0.1))
        x_start = max(0, x - int(w * 0.1))
        x_end = min(img.shape[1], x + w + int(w * 0.1))
        return img[y_start:y_end, x_start:x_end]
    return None

def predict(img):
    cropped = crop_head(img)
    if cropped is not None:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    gray = np.asarray(gray, dtype='float32') / 255.0
    gray = gray.reshape(1, 64, 64, 1)
    result = model(gray, training=False).numpy()[0]
    return float(result[0]*100), float(result[1]*100), cropped

# Capturar frame de cámara (índice 0 por defecto, cambialo si hace falta)
print("\nAbriendo cámara... (presioná ESPACIO para capturar, Q para salir)")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    con, sin, cropped = predict(frame)
    label = f"camara.py -> Con:{con:.1f}% Sin:{sin:.1f}%"
    color = (0,255,0) if con > sin else (0,0,255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Comparacion - camara directa", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord(' '):
        cv2.imwrite("comparacion_camara.jpg", frame)
        if cropped is not None:
            cv2.imwrite("comparacion_crop_camara.jpg", cropped)
        print(f"\n[CAMARA] Con casco: {con:.1f}% | Sin casco: {sin:.1f}%")
        
        # Leer qué guardó app.py
        if os.path.exists("debug_cropped_color.jpg"):
            web_crop = cv2.imread("debug_cropped_color.jpg")
            con_w, sin_w, _ = predict(web_crop)
            print(f"[WEB]    Con casco: {con_w:.1f}% | Sin casco: {sin_w:.1f}%")
            cv2.imshow("Web crop (lo que mandó el browser)", web_crop)
        else:
            print("[WEB] No hay debug_cropped_color.jpg todavía (abrí la app web primero)")

cap.release()
cv2.destroyAllWindows()
