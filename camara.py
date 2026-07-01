import keras
import cv2
import numpy as np
import glob

# Buscar todos los modelos en la carpeta
modelos_disponibles = glob.glob("*.keras")

if len(modelos_disponibles) == 0:
    print("No se encontraron modelos (.keras). ¡Entrena uno primero con ea.py!")
    exit()

print("--- MODELOS DISPONIBLES ---")
for i, modelo_path in enumerate(modelos_disponibles):
    print(f"[{i}] {modelo_path}")
print("---------------------------")

# Pedir al usuario que elija
seleccion = input(f"Ingresa el numero del modelo a cargar (0-{len(modelos_disponibles)-1}) [Por defecto: 0]: ")
if seleccion.strip() == "":
    seleccion = 0
else:
    try:
        seleccion = int(seleccion)
    except:
        print("Seleccion no valida. Usando el modelo 0.")
        seleccion = 0

modelo_elegido = modelos_disponibles[seleccion]

print(f"\nCargando '{modelo_elegido}'... (esto puede tomar unos segundos)")
try:
    model = keras.models.load_model(modelo_elegido)
except Exception as e:
    print(f"Error al cargar {modelo_elegido}.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def crop_head(img):
    h_orig, w_orig = img.shape[:2]
    max_dim = 500  # Tamaño máximo de detección para máxima velocidad
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
                
    # Filtrar caras en la mitad inferior (falsos positivos)
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
        return img[y_start:y_end, x_start:x_end], (x_start, y_start, x_end - x_start, y_end - y_start)
    return None, None

print("¡Modelo cargado exitosamente! Abriendo la camara...")

# Inicializar la camara (el numero 0 suele ser la webcam por defecto)
# Pedir al usuario que elija el índice de la cámara para permitir cambiar entre DroidCam y la integrada
print("--- CONFIGURACIÓN DE CÁMARA ---")
print("Si se abre una cámara incorrecta (ej. DroidCam), intenta usar otro número (ej. 1 o 2).")
seleccion_cam = input("Ingresa el número de la cámara a usar (0, 1, 2) [Por defecto: 0]: ")
if seleccion_cam.strip() == "":
    indice_cam = 0
else:
    try:
        indice_cam = int(seleccion_cam)
    except:
        print("Selección no válida. Usando la cámara 0.")
        indice_cam = 0

cap = cv2.VideoCapture(indice_cam)

if not cap.isOpened():
    print(f"Error: No se pudo acceder a la cámara con el índice {indice_cam}.")
    exit()

print("\n--- INSTRUCCIONES ---")
print("Presiona la tecla 'q' en la ventana del video para salir.")

while True:
    # Leer el frame actual de la camara
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la camara.")
        break
        
    cropped, bbox = crop_head(frame)
    
    if cropped is not None:
        gray_test = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.resize(gray_test, (64, 64))
        gray_test = np.asarray(gray_test, dtype='float32') / 255.0
        gray_test = gray_test.reshape(1, 64, 64, 1)
        
        result = model(gray_test, training=False).numpy()
        con = result[0][0] * 100
        sin = result[0][1] * 100
        
        if con > sin:
            label = f"Con casco: {con:.2f}%"
            color = (0, 255, 0) # Verde
        else:
            label = f"Sin casco: {sin:.2f}%"
            color = (0, 0, 255) # Rojo
            
        cx, cy, cw, ch = bbox
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), color, 2)
        cv2.putText(frame, label, (cx, max(cy - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, "Buscando rostro...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    # Mostrar la ventana
    cv2.imshow('Detector de Cascos en Tiempo Real', frame)
    
    # Salir si el usuario presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la camara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
