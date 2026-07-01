import keras
import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def crop_head(img):
    # Redimensionar si la imagen es muy grande para acelerar drásticamente la detección
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
        if len(faces_flipped) > 0:
            h_img, w_img = gray.shape
            faces = []
            for (xf, yf, wf, hf) in faces_flipped:
                real_x = w_img - (xf + wf)
                faces.append([real_x, yf, wf, hf])
                
    # Filtrar caras en la mitad inferior (falsos positivos)
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

model = keras.models.load_model("modelo.keras")

def predict_img(path):
    if not os.path.exists(path):
        return f"{path} no existe"
    img = cv2.imread(path)
    if img is None:
        return f"Error al leer {path}"
    
    cropped = crop_head(img)
    if cropped is not None:
        gray_test = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    gray_test = cv2.resize(gray_test, (64, 64))
    gray_test = np.asarray(gray_test, dtype='float32') / 255.0
    gray_test = gray_test.reshape(1, 64, 64, 1)
    result = model.predict(gray_test, verbose=0)
    con = result[0][0] * 100
    sin = result[0][1] * 100
    return f"{path} -> Con casco: {con:.2f}%, Sin casco: {sin:.2f}%"

print(predict_img("test.jpg"))
print(predict_img("con_casco/1.jpg"))
print(predict_img("con_casco/2.jpg"))
print(predict_img("sin_casco/1.jpg"))
