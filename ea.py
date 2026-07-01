import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import sys

cat=["con_casco" , "sin_casco"]
imgs=[]  
labels=[] 

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

x=0
for i in cat: 
    if not os.path.exists(i):
        os.makedirs(i)
    files = [f for f in os.listdir(i) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(files)
    print(f"\n[+] Cargando clase '{i}' ({total_files} imagenes)...")
    for idx, file in enumerate(files):
        print(f"\r    -> Procesando {idx+1}/{total_files}: {file}", end="", flush=True)
        img=cv2.imread(os.path.join(i, file))
        if img is not None:
            cropped = crop_head(img)
            if cropped is not None:
                gray_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img=cv2.resize(gray_img,(64,64)) 
            gray_img=np.asarray(gray_img)
            imgs.append(gray_img)
            labels.append(x) 
    print()
    x=x+1 

if len(imgs) == 0:
    print("\n[!] No se encontraron imagenes. Por favor, agrega algunas fotos a las carpetas 'con_casco' y 'sin_casco'.")
    sys.exit()


model=keras.Sequential([
    keras.layers.Conv2D(32, (3,3), input_shape=(64,64,1), activation="relu"), 
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(), 
    keras.layers.Dense(units=128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation="softmax") 
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
loss="sparse_categorical_crossentropy", 
metrics=["accuracy"])

imgs=np.array(imgs, dtype='float32') / 255.0 # Normalizar a [0, 1]
imgs = imgs.reshape(-1, 64, 64, 1) # Asegurar la dimension del canal para Data Augmentation
labels=np.array(labels)

# MAGIA: Data Augmentation para crear miles de variaciones de nuestras pocas fotos
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    horizontal_flip=True,
    zoom_range=0.08
)
datagen.fit(imgs)

# Entrenar usando el generador en lugar de los arreglos fijos
model.fit(datagen.flow(imgs, labels, batch_size=16), epochs=100)

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_modelo = f"modelo_{timestamp}.keras"

# Guarda una versión con fecha y hora
model.save(nombre_modelo)
# Guarda también la versión por defecto para los demás scripts
model.save("modelo.keras")

print(f"\n[+] Entrenamiento finalizado. Modelo guardado como '{nombre_modelo}' (y actualizado en 'modelo.keras')")

if not os.path.exists("test.jpg"):
    print("\n[!] Aviso: No se encontro 'test.jpg' para probar el modelo.")
else:
    test=cv2.imread("test.jpg") 
    if test is not None:
        cropped = crop_head(test)
        if cropped is not None:
            gray_test = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
            
        gray_test=cv2.resize(gray_test,(64,64))  
        gray_test=np.asarray(gray_test, dtype='float32') / 255.0
        gray_test=gray_test.reshape(1, 64, 64, 1) 

        result=model.predict(gray_test)
        print(result) 

        print("sin casco:" , result[0][1]*100, "%" )
        print("con casco:" , result[0][0]*100, "%" ) 
 



