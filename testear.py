import keras
import cv2
import numpy as np
import os
import glob

# Buscar todos los modelos en la carpeta
modelos_disponibles = glob.glob("*.keras")

if len(modelos_disponibles) == 0:
    print("No se encontraron modelos (.keras) en esta carpeta. ¡Entrena uno primero con ea.py!")
    exit()

print("--- MODELOS DISPONIBLES ---")
for i, modelo_path in enumerate(modelos_disponibles):
    print(f"[{i}] {modelo_path}")
print("---------------------------")

# Pedir al usuario que elija
seleccion = input(f"Ingresa el numero del modelo que quieres cargar (0-{len(modelos_disponibles)-1}) [Por defecto: 0]: ")
if seleccion.strip() == "":
    seleccion = 0
else:
    try:
        seleccion = int(seleccion)
    except:
        print("Seleccion no valida. Usando el modelo 0.")
        seleccion = 0

modelo_elegido = modelos_disponibles[seleccion]

print(f"\nCargando el cerebro de '{modelo_elegido}'... (esto puede tomar unos segundos)")
try:
    model = keras.models.load_model(modelo_elegido)
except Exception as e:
    print(f"Error al cargar {modelo_elegido}.")
    exit()

print("¡Modelo cargado exitosamente!\n")

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

while True:
    path = input("Ingresa el nombre o la ruta de la imagen a probar (o 'salir' para terminar): ")
    if path.lower() == 'salir':
        break
        
    if not os.path.exists(path):
        print(f" -> No se encontro el archivo: {path}\n")
        continue

    img = cv2.imread(path)
    if img is None:
        print(f" -> Error al leer la imagen {path}. Asegurate que sea JPG o PNG.\n")
        continue

    # Preprocesamiento
    cropped = crop_head(img)
    if cropped is not None:
        gray_test = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    gray_test = cv2.resize(gray_test, (64, 64))
    gray_test = np.asarray(gray_test, dtype='float32') / 255.0
    gray_test = gray_test.reshape(1, 64, 64, 1)
    
    # Prediccion
    result = model.predict(gray_test, verbose=0)
    con = result[0][0] * 100
    sin = result[0][1] * 100
    
    print("\n--- RESULTADOS ---")
    print(f"Probabilidad de CON casco: {con:.2f}%")
    print(f"Probabilidad de SIN casco: {sin:.2f}%")
    print("------------------\n")
