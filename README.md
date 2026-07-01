# Detector de Casco de Seguridad - Red Neuronal Convolucional (CNN)

Este proyecto implementa un sistema completo para entrenar, probar y ejecutar en tiempo real un **Detector de Casco de Seguridad** utilizando aprendizaje profundo (*Deep Learning*) con TensorFlow, Keras y OpenCV.

El sistema clasifica imágenes de personas en dos categorías principales:
1. **Con Casco** (`con_casco`)
2. **Sin Casco** (`sin_casco`)

---

## 📂 Estructura del Proyecto

A continuación se detalla la función de cada archivo en este repositorio:

*   **Directorios de Datos:**
    *   `con_casco/`: Carpeta que almacena las imágenes de entrenamiento para la clase con casco.
    *   `sin_casco/`: Carpeta que almacena las imágenes de entrenamiento para la clase sin casco.
*   **Scripts de Preparación:**
    *   [convert.py](file:///c:/Users/Alex/Desktop/ae/convert.py): Convierte imágenes de formato crudo `.dng` a formato estándar `.jpg` y elimina el archivo original.
    *   [renombrar.py](file:///c:/Users/Alex/Desktop/ae/renombrar.py): Estandariza el dataset renombrando secuencialmente las imágenes en ambas carpetas (`1.jpg`, `2.jpg`, etc.) y unificando extensiones.
*   **Script de Entrenamiento:**
    *   [ea.py](file:///c:/Users/Alex/Desktop/ae/ea.py): Carga el dataset, aplica *Data Augmentation*, define la arquitectura de la red neuronal convolucional (CNN), entrena el modelo durante 100 épocas y lo guarda en formato `.keras`.
*   **Scripts de Prueba e Inferencia:**
    *   [camara.py](file:///c:/Users/Alex/Desktop/ae/camara.py): Permite seleccionar un modelo entrenado y activa la cámara web para realizar detección en tiempo real con una superposición gráfica verde (Con casco) o roja (Sin casco).
    *   [testear.py](file:///c:/Users/Alex/Desktop/ae/testear.py): Script interactivo en línea de comandos para probar imágenes individuales ingresando su ruta y seleccionando un modelo específico.
    *   [check.py](file:///c:/Users/Alex/Desktop/ae/check.py): Realiza una prueba rápida y estática de predicción sobre imágenes predefinidas (`test.jpg`, muestras locales) usando el modelo por defecto `modelo.keras`.

---

## 🛠️ Requisitos y Dependencias

Para poder ejecutar los scripts de este proyecto, es necesario tener instalado Python 3.x y las siguientes librerías:

```bash
pip install tensorflow keras opencv-python numpy rawpy imageio
```

---

## 🚀 Flujo de Trabajo y Uso

### 1. Preparación del Dataset
Antes de entrenar el modelo, es necesario recopilar imágenes y estructurarlas en las carpetas correspondientes:

1. Guarda las fotos con casco en la carpeta `con_casco/`.
2. Guarda las fotos sin casco en la carpeta `sin_casco/`.
3. Si has tomado las fotos en formato RAW (`.dng`), puedes convertirlas automáticamente ejecutando:
   ```bash
   python convert.py
   ```
4. Para tener un dataset organizado y evitar colisiones de nombres, renombra y secuencia todas las imágenes ejecutando:
   ```bash
   python renombrar.py
   ```
   *Esto convertirá las extensiones `.jpeg` a `.jpg` y renombrará las fotos secuencialmente de `1.jpg` en adelante.*

### 2. Entrenamiento del Modelo
El entrenamiento utiliza una red neuronal convolucional (CNN) optimizada para imágenes en escala de grises de `64x64` píxeles. También se implementa *Data Augmentation* para robustecer el modelo (rotación, desplazamientos, volteos y zoom).

Para entrenar el modelo ejecuta:
```bash
python ea.py
```
*   Al finalizar el entrenamiento (100 épocas), se generarán dos archivos:
    *   Un archivo con marca de tiempo: `modelo_YYYYMMDD_HHMMSS.keras` (ideal para llevar un historial).
    *   El modelo por defecto actualizado: `modelo.keras`.

### 3. Pruebas y Clasificación Manual
Si deseas verificar la precisión del modelo en imágenes estáticas de prueba:

*   **Prueba rápida:**
    ```bash
    python check.py
    ```
    *Evalúa imágenes fijas como `test.jpg` y unas muestras básicas para validar que el modelo cargue correctamente.*

*   **Prueba interactiva:**
    ```bash
    python testear.py
    ```
    *Te permitirá elegir cuál de los modelos guardados deseas cargar, y posteriormente podrás ingresar la ruta de cualquier imagen (por ejemplo, `test.jpg`) para ver la probabilidad detallada en consola.*

### 4. Detección en Tiempo Real (Webcam)
Para desplegar la aplicación interactiva de detección utilizando tu cámara web:

```bash
python camara.py
```
1. El script te listará todos los modelos `.keras` disponibles en el directorio y te pedirá elegir uno.
2. Se abrirá una ventana de video de tu cámara web.
3. El detector convertirá el stream a escala de grises en tiempo real y clasificará si llevas casco o no:
    *   **Con casco:** Texto verde sobre la pantalla indicando la certeza.
    *   **Sin casco:** Texto rojo sobre la pantalla indicando la certeza.
4. Presiona la tecla **`q`** en la ventana de video para cerrar la cámara y finalizar el script.

---

## 🧠 Arquitectura de la Red Neuronal

La red convolucional implementada en [ea.py](file:///c:/Users/Alex/Desktop/ae/ea.py) consta de la siguiente estructura secuencial:

1.  **Conv2D:** 32 filtros de tamaño `3x3`, función de activación *ReLU*, esperando entrada de `64x64` píxeles en 1 canal (escala de grises).
2.  **MaxPooling2D:** Reducción de dimensiones espaciales con pool de `2x2`.
3.  **Conv2D:** 64 filtros de tamaño `3x3`, función de activación *ReLU*.
4.  **MaxPooling2D:** Reducción de dimensiones con pool de `2x2`.
5.  **Dropout (50%):** Regularización para evitar el sobreajuste (*overfitting*).
6.  **Flatten:** Aplanado de las características en un vector unidimensional.
7.  **Dense:** Capa completamente conectada con 100 neuronas y activación *ReLU*.
8.  **Dense (Salida):** 2 neuronas con función de activación *Softmax* para obtener la distribución de probabilidad de las clases (`con_casco` y `sin_casco`).
