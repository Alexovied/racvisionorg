# =============================================================================
# ea.py — ENTRENAMIENTO DE LA RED NEURONAL
# =============================================================================
# Este script hace UNA SOLA COSA: enseñarle a la IA a distinguir entre
# una persona con casco y una persona sin casco.
#
# Flujo general:
#   1. Lee las fotos del dataset (carpetas con_casco/ y sin_casco/)
#   2. Detecta la cara en cada foto y recorta el área de la cabeza
#   3. Construye la arquitectura de la red neuronal
#   4. Entrena la red con todas las fotos
#   5. Guarda el modelo entrenado en un archivo .keras
# =============================================================================

# ── Librerías necesarias ─────────────────────────────────────────────────────
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # para Data Augmentation
import numpy as np      # manejo de arrays numéricos
import cv2              # OpenCV: procesamiento de imágenes
import os               # para leer archivos y carpetas
import sys              # para salir del script si hay error


# =============================================================================
# PASO 0 — Configuración inicial
# =============================================================================

# Las dos clases que la IA tiene que aprender a distinguir.
# El ORDEN importa: índice 0 = con_casco, índice 1 = sin_casco
cat = ["con_casco", "sin_casco"]

# Acá se van a guardar todas las imágenes procesadas y sus etiquetas
imgs   = []   # lista de imágenes (matrices de píxeles)
labels = []   # lista de etiquetas: 0 = con casco, 1 = sin casco

# =============================================================================
# PASO 1 — Detector de caras (Haar Cascade)
# =============================================================================
# El Haar Cascade es un algoritmo clásico de visión computarizada (del año 2001)
# que detecta caras en imágenes. Lo usamos ANTES de pasarle la foto a la IA
# para recortar solo el área relevante: la cabeza (donde está el casco).
#
# Usamos DOS tipos de detector:
#   - frontal: cara mirando de frente a la cámara
#   - perfil:  cara de costado
face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


# =============================================================================
# PASO 2 — Función crop_head: recortar el área de la cabeza
# =============================================================================
def crop_head(img):
    """
    Recibe una imagen completa y devuelve un recorte que incluye
    la cara + el área del casco por encima.

    Estrategia de detección (en orden de prioridad):
      1. Cara frontal
      2. Cara de perfil
      3. Cara de perfil en imagen espejada (perfil mirando al otro lado)
    """

    # Redimensionar si la imagen es muy grande → acelera la detección
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Convertir a escala de grises para que el detector funcione
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Intento 1: cara frontal
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

    # Intento 2: si no encontró, prueba con cara de perfil
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

    # Intento 3: espejamos la imagen y buscamos de nuevo
    # Esto cubre el caso de perfil mirando hacia el lado derecho
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
        if len(faces_flipped) > 0:
            h_img, w_img = gray.shape
            faces = []
            for (xf, yf, wf, hf) in faces_flipped:
                # Convertir coordenadas de la imagen espejada a la original
                real_x = w_img - (xf + wf)
                faces.append([real_x, yf, wf, hf])

    # Filtrar falsos positivos: ignorar "caras" detectadas en la mitad inferior
    # de la imagen (probablemente ropa o fondo, no una cara real)
    valid_faces = [f for f in faces if f[1] + f[3]/2 < img.shape[0] * 0.70]

    if len(valid_faces) > 0:
        # Si hay varias caras detectadas, quedarse con la más grande
        valid_faces = sorted(valid_faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = valid_faces[0]

        # ── Clave del sistema: el margen superior ────────────────────────────
        # El casco está POR ENCIMA de la cara. Si recortamos solo la cara,
        # el casco queda fuera del frame y la IA no puede verlo.
        # Por eso extendemos 1.2x la altura de la cara hacia arriba.
        #
        #   ┌──────────────────┐  ← y_start (1.2x por encima de la cara)
        #   │   [ C A S C O ]  │
        #   │   ┌──────────┐   │
        #   │   │  CARA    │   │  ← área detectada por Haar Cascade
        #   │   └──────────┘   │
        #   └──────────────────┘  ← y_end (0.15x por debajo de la cara)
        y_start = max(0, y - int(h * 1.2))    # margen superior: 1.2x altura de cara
        y_end   = min(img.shape[0], y + h + int(h * 0.15))  # margen inferior: 0.15x
        x_start = max(0, x - int(w * 0.15))   # margen lateral izquierdo
        x_end   = min(img.shape[1], x + w + int(w * 0.15))  # margen lateral derecho

        return img[y_start:y_end, x_start:x_end]

    # Si no se detectó ninguna cara válida, devolver None
    # → la imagen se saltea durante el entrenamiento (evita ruido)
    return None


# =============================================================================
# PASO 3 — Cargar y procesar el dataset
# =============================================================================
# Recorremos ambas carpetas (con_casco/ y sin_casco/) y procesamos cada foto.
# Solo se incluyen las fotos donde se detectó una cara → garantiza calidad.

x = 0  # contador de clase: 0 = con_casco, 1 = sin_casco

for i in cat:
    if not os.path.exists(i):
        os.makedirs(i)

    files = [f for f in os.listdir(i) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(files)
    print(f"\n[+] Cargando clase '{i}' ({total_files} imagenes)...")

    for idx, file in enumerate(files):
        print(f"\r    -> Procesando {idx+1}/{total_files}: {file}", end="", flush=True)
        img = cv2.imread(os.path.join(i, file))

        if img is not None:
            cropped = crop_head(img)

            if cropped is not None:
                # Foto válida: se detectó una cara y se recortó el área de la cabeza
                rgb_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                rgb_img = cv2.resize(rgb_img, (128, 128))

                rgb_img = np.asarray(rgb_img)
                imgs.append(rgb_img)
                labels.append(x)   # x=0 → con casco, x=1 → sin casco
            else:
                # No se detectó cara → saltear esta foto
                # IMPORTANTE: incluirla sin recortar sería ruido puro:
                # la IA aprendería a clasificar el fondo en lugar del casco
                print(f" [SKIP: sin cara detectable]", end="", flush=True)

    print()
    x = x + 1  # pasar a la siguiente clase

if len(imgs) == 0:
    print("\n[!] No se encontraron imagenes. Por favor, agrega algunas fotos a las carpetas 'con_casco' y 'sin_casco'.")
    sys.exit()


# =============================================================================
# PASO 4 — Construir la arquitectura de la red neuronal (Transfer Learning)
# =============================================================================

# Cargar el modelo base pre-entrenado (MobileNetV2)
base_model = keras.applications.MobileNetV2(
    weights='imagenet',       # Pesos pre-entrenados en millones de imágenes
    include_top=False,        # Quitar la última capa original
    input_shape=(128, 128, 3) # Entrada a color (RGB) 128x128
)

# Congelar el modelo base (no entrenar sus pesos, ya saben "ver")
base_model.trainable = False

# Crear nuestro modelo final
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])


# =============================================================================
# PASO 5 — Compilar el modelo
# =============================================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =============================================================================
# PASO 6 — Preparar los datos para el entrenamiento
# =============================================================================

# Preparación con preprocess_input de MobileNetV2 para imágenes a color
imgs = np.array(imgs, dtype='float32')
imgs = keras.applications.mobilenet_v2.preprocess_input(imgs)

labels = np.array(labels)


# =============================================================================
# PASO 7 — Data Augmentation agresivo
# =============================================================================
# Más variaciones → más resistencia a condiciones reales de webcam.
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    horizontal_flip=True,
    zoom_range=0.20,
    brightness_range=[0.6, 1.4],   # simula webcam con luz variable
    shear_range=0.10,
    fill_mode='nearest'
)
datagen.fit(imgs)


# =============================================================================
# PASO 7.5 — Separar datos de validación
# =============================================================================
from sklearn.model_selection import train_test_split

imgs_train, imgs_val, labels_train, labels_val = train_test_split(
    imgs, labels, test_size=0.20, random_state=42, stratify=labels
)

print(f"[+] Dataset: {len(imgs_train)} entrenamiento, {len(imgs_val)} validación")


# =============================================================================
# PASO 8 — Callbacks: EarlyStopping + ReduceLROnPlateau
# =============================================================================
callbacks_fase1 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]


# =============================================================================
# PASO 9 — FASE 1: Entrenar solo las capas Dense (base congelado)
# =============================================================================
print("\n" + "="*60)
print("[+] FASE 1: Entrenando capas Dense (base congelado)...")
print("="*60 + "\n")

model.fit(
    datagen.flow(imgs_train, labels_train, batch_size=16),
    epochs=30,
    validation_data=(imgs_val, labels_val),
    callbacks=callbacks_fase1
)


# =============================================================================
# PASO 10 — FASE 2: Fine-Tuning
# =============================================================================
# Descongelar las últimas 30 capas del base model para que se adapten
# a nuestro dominio específico (cascos de seguridad).
print("\n" + "="*60)
print("[+] FASE 2: Fine-Tuning — descongelando últimas 30 capas de MobileNetV2...")
print("="*60 + "\n")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"    Capas descongeladas: {trainable_count} de {len(base_model.layers)}")

# ⚠️ Learning rate MUY bajo para fine-tuning → no destruir pesos pre-entrenados
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_fase2 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-8,
        verbose=1
    )
]

model.fit(
    datagen.flow(imgs_train, labels_train, batch_size=16),
    epochs=50,
    validation_data=(imgs_val, labels_val),
    callbacks=callbacks_fase2
)


# =============================================================================
# PASO 11 — Guardar el modelo entrenado
# =============================================================================
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_modelo = f"modelo_{timestamp}.keras"

model.save(nombre_modelo)
model.save("modelo.keras")

print(f"\n[+] Entrenamiento finalizado.")
print(f"    Guardado como: {nombre_modelo}")
print(f"    Actualizado:   modelo.keras")


# =============================================================================
# PASO 12 — Prueba rápida con test.jpg
# =============================================================================
if not os.path.exists("test.jpg"):
    print("\n[!] Aviso: No se encontro 'test.jpg' para probar el modelo.")
else:
    test_img = cv2.imread("test.jpg")
    if test_img is not None:
        cropped = crop_head(test_img)  # devuelve RGB en color
        if cropped is not None:
            rgb_test = cv2.resize(cropped, (128, 128))
        else:
            rgb_test = cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), (128, 128))

        rgb_test = np.asarray(rgb_test, dtype='float32')
        rgb_test = keras.applications.mobilenet_v2.preprocess_input(rgb_test)
        rgb_test = rgb_test.reshape(1, 128, 128, 3)

        result = model.predict(rgb_test)
        print(f"\n[TEST] con_casco: {result[0][0]*100:.1f}%  |  sin_casco: {result[0][1]*100:.1f}%")
