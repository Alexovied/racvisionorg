# RAC Vision: control de EPP con cámaras

## 🎬 Video demo

▶️ **[Ver el video demo de RAC Vision](PEGAR_ENLACE_DE_YOUTUBE_AQUI)** · 🌐 Demo en vivo: https://racvision.org

<img src="docs/actividades/qr-racvision.png" alt="Código QR a racvision.org" width="140">

RAC Vision detecta en tiempo real si las personas llevan casco y chaleco, reconoce caras, mide la postura y avisa (panel web, Telegram, MQTT y la caja con ESP32) cuando falta un EPP.

---

## 📂 Estructura

Todo el código de la app está en `src/`:

*   **App web (Flask):** `app.py` (detector público y `/predict`), `admin.py` (panel, usuarios, reportes), `data_store.py` (datos y configuración), `templates/`, `static/`.
*   **Motor de detección:** `motor.py`, `helmet_detector.py` (casco), `epp.py` (chaleco), `pose.py` y `cuerpo.py` (postura), `face_detector.py` y `face_id.py` (caras), `riesgo.py`, `salud.py`.
*   **Avisos, hardware y wearables:** `aviso_telegram.py`, `aviso_mqtt.py`, `hardware.py`, `wearables.py`, `simulador_wearable.py`, `aprendizaje.py`. El firmware de la caja está en `firmware/`.
*   **SGEP-Dashboard:** `sgep.py`, `sgep_seed.py`, `sgep_local.py` (corre solo el dashboard en localhost: `python src/sgep_local.py`).
*   **Modelos (`modelos_app/`):** `detector_casco.onnx`, `detector_chaleco.onnx`, `detector_pose.onnx`, los `*_int8.*` de OpenVINO, los dos de caras (`face_*.onnx`) y `modelo.keras` (clasificador viejo, se puede elegir en la configuración).
*   **Entrenamiento y medición (`herramientas/`, se corren desde la raíz, ej. `python herramientas/evaluar_modelos.py`):** `ea.py` y `retrain.py` (modelo Keras), `cuantizar.py`, `evaluar_modelos.py`, `benchmark_inferencia.py`, `benchmark_carga.py`, `capturar_casco.py`, `relay_camara.py` (reenvía una cámara de otra red al servidor). Los notebooks de Colab (entrenamiento de los detectores y clips de video) están en `entrenamiento/`. Las fotos locales (`con_casco`, `sin_casco`, `personas`) van en `datasets/`, que no se sube a git.
*   **Revisar alertas:** `revision.py`: corregir a mano lo que vio el modelo y descargar el dataset YOLO para reentrenar.
*   **Tests y docs:** `tests/`, `docs/`.

---

## 🚀 Uso

```bash
pip install -r requirements-dev.txt
python -m pytest tests
gunicorn --chdir src --worker-class gthread --workers 1 --threads 32 --timeout 120 app:app
```

Siempre **un solo worker con hilos**: el estado del motor, los wearables y el candado de `data/` viven en memoria del proceso, y cada video en vivo (MJPEG) ocupa un hilo mientras está abierto. Con el worker `sync` por defecto, abrir Monitoreo deja afuera a los cuadros de las cámaras relay. El `CMD` del Dockerfile del servidor tiene que usar el mismo comando.
