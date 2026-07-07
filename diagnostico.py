"""
diagnostico.py — Corre TODOS los crops de entrenamiento contra el modelo actual
y muestra cuántos clasifica bien y MAL por clase.
"""
import keras
import cv2
import numpy as np
import os

MODEL_PATH = "modelo_20260707_022548.keras"

print(f"[+] Cargando {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)
print("[+] Modelo listo.\n")

face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# ⚠️ MISMO crop que ea.py y app.py actualizado (margen 1.2x)
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
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
        if len(faces_flipped) > 0:
            h_img, w_img = gray.shape
            faces = []
            for (xf, yf, wf, hf) in faces_flipped:
                real_x = w_img - (xf + wf)
                faces.append([real_x, yf, wf, hf])

    valid_faces = [f for f in faces if f[1] + f[3]/2 < img.shape[0] * 0.70]
    if len(valid_faces) > 0:
        valid_faces = sorted(valid_faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = valid_faces[0]
        # Mismo margen 1.2x que en ea.py
        y_start = max(0, y - int(h * 1.2))
        y_end   = min(img.shape[0], y + h + int(h * 0.15))
        x_start = max(0, x - int(w * 0.15))
        x_end   = min(img.shape[1], x + w + int(w * 0.15))
        return img[y_start:y_end, x_start:x_end]
    return None

def predecir(path):
    img = cv2.imread(path)
    if img is None:
        return None, None
    cropped = crop_head(img)
    src = cropped if cropped is not None else img
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    gray = gray.astype('float32') / 255.0
    gray = gray.reshape(1, 64, 64, 1)
    result = model.predict(gray, verbose=0)
    con = float(result[0][0]) * 100
    sin = float(result[0][1]) * 100
    return con, sin

# ---- Correr por cada clase ----
clases = {
    "con_casco": ("con_casco", 0),  # label=0 es con_casco
    "sin_casco": ("sin_casco", 1),
}

total_ok  = 0
total_err = 0

for clase_nombre, (carpeta, label_esperado) in clases.items():
    if not os.path.exists(carpeta):
        print(f"[!] No se encontró carpeta: {carpeta}")
        continue

    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    ok = 0; err = 0
    errores = []

    for fname in sorted(archivos):
        fpath = os.path.join(carpeta, fname)
        con, sin = predecir(fpath)
        if con is None:
            continue
        pred = 0 if con > sin else 1  # 0=con_casco, 1=sin_casco
        if pred == label_esperado:
            ok += 1
        else:
            err += 1
            errores.append((fname, con, sin))

    total = ok + err
    pct = (ok/total*100) if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"  Clase: {clase_nombre}  ({total} imágenes)")
    print(f"  [OK]  Correctas: {ok}  ({pct:.1f}%)")
    print(f"  [ERR] Errores:   {err}  ({100-pct:.1f}%)")
    if errores:
        print(f"\n  Imagenes mal clasificadas:")
        for fname, con, sin in errores[:10]:
            print(f"    {fname}: con={con:.1f}% sin={sin:.1f}%")
    total_ok  += ok
    total_err += err

print(f"\n{'='*50}")
print(f"TOTAL — Correctas: {total_ok}  Errores: {total_err}  ({total_ok/(total_ok+total_err)*100:.1f}% accuracy)")
print('='*50)
