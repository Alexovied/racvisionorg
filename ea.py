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

                # Convertir a escala de grises
                # ¿Por qué gris? El color del casco varía (amarillo, blanco, azul...)
                # Lo que importa es la FORMA, no el color → gris es suficiente
                gray_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                # Redimensionar a 64x64 píxeles
                # Todas las fotos tienen que tener el mismo tamaño para entrar a la red
                gray_img = cv2.resize(gray_img, (64, 64))

                gray_img = np.asarray(gray_img)
                imgs.append(gray_img)
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
# PASO 4 — Construir la arquitectura de la red neuronal (CNN)
# =============================================================================
# Una CNN (Convolutional Neural Network) es el tipo de red neuronal estándar
# para procesar imágenes. Funciona aplicando filtros que detectan patrones
# de menor a mayor complejidad:
#
#   Capa 1 (Conv 32): detecta bordes y texturas básicas
#   Capa 2 (Conv 64): detecta formas simples (curvas, esquinas)
#   Capa 3 (Conv 128): detecta patrones complejos ("objeto redondeado arriba")
#   Dense(128): combina todas las características detectadas
#   Dense(2): decide: ¿con casco o sin casco?
#
# Sequential = capas apiladas en secuencia, la salida de una entra a la siguiente

model = keras.Sequential([

    # ── BLOQUE CONVOLUCIONAL 1 ──────────────────────────────────────────────
    # Aplica 32 filtros de 3x3 sobre la imagen de 64x64
    # Cada filtro detecta un patrón diferente (borde vertical, horizontal, etc.)
    keras.layers.Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation="relu"),
    # BatchNorm: normaliza los valores → entrenamiento más estable
    keras.layers.BatchNormalization(),
    # MaxPooling: reduce el tamaño a la mitad (64x64 → 32x32)
    # Se queda con el valor máximo de cada bloque 2x2 → más robusto
    keras.layers.MaxPooling2D(2, 2),

    # ── BLOQUE CONVOLUCIONAL 2 ──────────────────────────────────────────────
    # 64 filtros → detecta combinaciones de los patrones anteriores (formas)
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),   # 32x32 → 16x16

    # ── BLOQUE CONVOLUCIONAL 3 ──────────────────────────────────────────────
    # 128 filtros → detecta patrones de alto nivel ("casco sobre la cabeza")
    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),   # 16x16 → 7x7

    # ── TRANSICIÓN A DECISIÓN ───────────────────────────────────────────────
    # Flatten: convierte el mapa de características 3D en un vector 1D
    # Es el puente entre la parte visual y la parte de decisión
    keras.layers.Flatten(),

    # ── CAPA DE DECISIÓN ────────────────────────────────────────────────────
    # 128 neuronas, cada una conectada a todas las anteriores
    # Combina todas las características para tomar la decisión final
    keras.layers.Dense(units=128, activation="relu"),
    keras.layers.BatchNormalization(),

    # Dropout(0.5): durante el entrenamiento apaga el 50% de neuronas al azar
    # Fuerza al modelo a no depender de neuronas individuales → generaliza mejor
    # En predicción se desactiva automáticamente (usa todas las neuronas)
    keras.layers.Dropout(0.5),

    # ── CAPA DE SALIDA ──────────────────────────────────────────────────────
    # 2 neuronas = 2 clases (con casco, sin casco)
    # Softmax convierte los números en probabilidades que suman exactamente 100%
    # Ejemplo de salida: [0.9999, 0.0001] → 99.99% con casco, 0.01% sin casco
    keras.layers.Dense(2, activation="softmax")
])


# =============================================================================
# PASO 5 — Compilar el modelo (configurar cómo va a aprender)
# =============================================================================
model.compile(
    # Adam: optimizador que ajusta automáticamente la velocidad de aprendizaje
    # lr=0.0005 es más conservador que el default (0.001) → aprende más despacio
    # pero con más precisión, menos riesgo de "pasarse" del punto óptimo
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),

    # La función de pérdida mide qué tan equivocada está la red en cada predicción
    # sparse_categorical_crossentropy es la estándar para clasificación con clases enteras
    loss="sparse_categorical_crossentropy",

    # Métrica a monitorear durante el entrenamiento (solo informativa, no afecta el aprendizaje)
    metrics=["accuracy"]
)


# =============================================================================
# PASO 6 — Preparar los datos para el entrenamiento
# =============================================================================

# Normalizar: convertir píxeles de rango [0, 255] a [0.0, 1.0]
# Las redes neuronales trabajan mejor con valores pequeños y uniformes
imgs = np.array(imgs, dtype='float32') / 255.0

# Agregar dimensión del canal: (N, 64, 64) → (N, 64, 64, 1)
# La CNN espera imágenes con dimensión de canal explícita (1 = escala de grises)
imgs = imgs.reshape(-1, 64, 64, 1)

labels = np.array(labels)


# =============================================================================
# PASO 7 — Data Augmentation
# =============================================================================
# PROBLEMA: ~190 fotos son muy pocas para entrenar una CNN.
# Con tan pocos datos el modelo memoriza las fotos en lugar de aprender
# el concepto "casco" → falla con imágenes nuevas (sobreajuste / overfitting).
#
# SOLUCIÓN: Data Augmentation. En cada epoch, cada imagen se transforma
# aleatoriamente dentro de estos rangos → el modelo "ve" millones de
# variaciones distintas sin necesitar más fotos reales.
#
# Parámetros conservadores (funciona bien con dataset pequeño):
datagen = ImageDataGenerator(
    rotation_range=15,         # rota la imagen hasta ±15 grados
    width_shift_range=0.08,    # desplaza horizontalmente hasta ±8%
    height_shift_range=0.08,   # desplaza verticalmente hasta ±8%
    horizontal_flip=True,      # espeja la imagen (izq ↔ der) aleatoriamente
    zoom_range=0.10            # hace zoom ±10% → simula diferentes distancias
)
datagen.fit(imgs)


# =============================================================================
# PASO 8 — ENTRENAMIENTO
# =============================================================================
# Acá es donde la IA realmente "aprende".
#
# ¿Cómo funciona internamente?
#   1. Le pasa un batch de 16 imágenes con sus etiquetas
#   2. La red predice: "creo que esto es con/sin casco"
#   3. Se calcula el error (loss) entre la predicción y la respuesta correcta
#   4. El algoritmo de backpropagation calcula en qué dirección ajustar cada peso
#   5. Adam actualiza todos los millones de parámetros de la red
#   6. Se repite con el siguiente batch
#   7. Cuando recorrió todo el dataset = 1 epoch completado
#   8. Se hace esto 120 veces (120 epochs)
#
# epochs=120: 120 pasadas completas por todo el dataset
# batch_size=16: aprende de 16 imágenes a la vez antes de actualizar los pesos

print("\n[+] Iniciando entrenamiento... (esto puede tardar unos minutos)\n")
model.fit(datagen.flow(imgs, labels, batch_size=16), epochs=120)


# =============================================================================
# PASO 9 — Guardar el modelo entrenado
# =============================================================================
# El archivo .keras contiene TODO lo que la red aprendió:
#   - La arquitectura (qué capas tiene y cómo están conectadas)
#   - Los pesos (los millones de parámetros ajustados durante el entrenamiento)
#
# Una vez guardado, app.py puede cargarlo y usarlo para hacer predicciones
# en milisegundos, sin necesidad de reentrenar.

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_modelo = f"modelo_{timestamp}.keras"

# Versión con timestamp → permite conservar historial de modelos entrenados
model.save(nombre_modelo)
# Versión genérica → la que usan app.py, camara.py y testear.py por defecto
model.save("modelo.keras")

print(f"\n[+] Entrenamiento finalizado. Modelo guardado como '{nombre_modelo}' (y actualizado en 'modelo.keras')")


# =============================================================================
# PASO 10 — Prueba rápida con test.jpg
# =============================================================================
# Al terminar el entrenamiento, el script prueba el modelo recién entrenado
# con una imagen de referencia (test.jpg) para verificar que funciona.
# Este es el mismo proceso que hace app.py en tiempo real, pero con una sola foto.

if not os.path.exists("test.jpg"):
    print("\n[!] Aviso: No se encontro 'test.jpg' para probar el modelo.")
else:
    test = cv2.imread("test.jpg")
    if test is not None:
        # Mismo preprocesado que en el entrenamiento
        cropped = crop_head(test)
        if cropped is not None:
            gray_test = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

        gray_test = cv2.resize(gray_test, (64, 64))
        gray_test = np.asarray(gray_test, dtype='float32') / 255.0
        gray_test = gray_test.reshape(1, 64, 64, 1)   # batch de 1 imagen

        # Inferencia: el modelo predice
        result = model.predict(gray_test)
        print(result)

        # Mostrar resultado legible
        # result[0][0] = probabilidad de con_casco
        # result[0][1] = probabilidad de sin_casco
        print("sin casco:", result[0][1] * 100, "%")
        print("con casco:", result[0][0] * 100, "%")
