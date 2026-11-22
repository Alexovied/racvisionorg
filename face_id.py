"""Reconocimiento de personas con SFace (OpenCV).

Alta: desde el panel (Personas) o con `python face_id.py enrolar`, que lee
personas/<nombre>/ con 5-10 fotos de frente. Se calcula un embedding por foto y
se guarda en data/personas.json (solo números, nunca las fotos). Quién la
registró y cuándo queda en data/personas_registro.json.

En /predict, `identificar(img, bbox)` busca la cara dentro de la cabeza
detectada y devuelve el nombre más parecido, "Desconocido" si nadie supera el
umbral, o None si no se ve la cara (de espaldas, tapada).
"""
import glob
import json
import os
import sys
import threading
from datetime import datetime

import cv2
import numpy as np

import data_store
import face_detector
import helmet_detector  # Core y dispositivo de OpenVINO

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "face_recognition_sface_2021dec.onnx")

# Similitud coseno mínima para dar un nombre. OpenCV sugiere 0.363, pero en las
# fotos del proyecto una misma persona dio 0.54-0.77 y personas distintas hasta
# 0.37: con 0.45 se prefiere "Desconocido" antes que un nombre equivocado.
UMBRAL = 0.45
MIN_FOTOS = 3   # con menos fotos útiles el reconocimiento es poco confiable

_sface = None
if os.path.exists(MODEL_PATH):
    try:
        _sface = cv2.FaceRecognizerSF.create(MODEL_PATH, "")
    except cv2.error as e:
        print(f"[!] No se pudo cargar SFace ({e})")

# La red de SFace corre con OpenVINO; alignCrop (alinear la cara) sigue en OpenCV
_sface_ov, _sface_lock = None, threading.Lock()
if _sface is not None and helmet_detector._core is not None:
    try:
        _sface_ov = helmet_detector._core.compile_model(MODEL_PATH, helmet_detector.DEVICE)
    except Exception as e:
        print(f"[!] SFace sigue con cv2.dnn ({e})")


def _feature(alineada):
    if _sface_ov is None:
        return _sface.feature(alineada)
    # Mismo preprocesamiento que FaceRecognizerSF.feature: 112x112, RGB, sin escalar
    blob = cv2.dnn.blobFromImage(alineada, 1, (112, 112), (0, 0, 0), swapRB=True)
    with _sface_lock:
        return _sface_ov(blob)[0].copy()

_cache = {"mtime": None, "personas": []}


def _ruta_db():
    return os.path.join(data_store.DATA_DIR, "personas.json")


CLAVE_ENV = "RAC_CLAVE_BIOMETRIA"   # clave Fernet: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"


def _fernet():
    clave = os.environ.get(CLAVE_ENV)
    if not clave:
        return None
    from cryptography.fernet import Fernet
    return Fernet(clave.encode())


def _leer_db(estricto=False):
    """{nombre: [embeddings]}. Con RAC_CLAVE_BIOMETRIA el archivo está cifrado en reposo (Ley 25.326:
    los datos biométricos son sensibles); un archivo viejo sin cifrar se lee igual y se cifra al guardar."""
    datos = data_store._load_json("personas.json", {})
    if not (isinstance(datos, dict) and set(datos) == {"cifrado"} and isinstance(datos["cifrado"], str)):
        return datos
    try:
        return json.loads(_fernet().decrypt(datos["cifrado"].encode()))
    except Exception as e:  # sin clave o clave equivocada: nadie se reconoce, pero no se borra nada
        print(f"[!] personas.json está cifrado y no se pudo abrir ({type(e).__name__}): revisá {CLAVE_ENV}", flush=True)
        if estricto:   # quien va a escribir no puede seguir: pisaría a todas las personas con un archivo nuevo
            raise RuntimeError(f"personas.json está cifrado y falta o no coincide {CLAVE_ENV}") from e
        return {}


def _escribir_db(datos):
    f = _fernet()
    data_store._save_json("personas.json", {"cifrado": f.encrypt(json.dumps(datos).encode()).decode()} if f else datos)


def _guardar(datos, registro):
    _escribir_db(datos)
    data_store._save_json("personas_registro.json", registro)
    _cache["mtime"] = None  # recargar aunque el mtime no haya cambiado


def _embeddings(imagenes):
    """[embedding] de las fotos con una cara visible y cuántas no la tenían."""
    embs, sin_cara = [], 0
    for img in imagenes:
        s = min(1.0, 1280 / max(img.shape[:2]))
        emb = embedding(cv2.resize(img, None, fx=s, fy=s) if s < 1 else img)
        if emb is None:
            sin_cara += 1
        else:
            embs.append(emb.tolist())
    return embs, sin_cara


def listar():
    """[{nombre, fotos, alta, por}] de las personas registradas."""
    datos = _leer_db()
    registro = data_store._load_json("personas_registro.json", {})
    return [{"nombre": n, "fotos": len(e), **registro.get(n, {})} for n, e in sorted(datos.items())]


def registrar(nombre, imagenes, por):
    """Agrega a una persona con sus fotos (ya decodificadas). Devuelve (útiles, sin cara);
    solo guarda si hay al menos MIN_FOTOS útiles. Las fotos no se guardan."""
    embs, sin_cara = _embeddings(imagenes)
    if len(embs) >= MIN_FOTOS:
        datos = _leer_db(estricto=True)
        registro = data_store._load_json("personas_registro.json", {})
        datos[nombre] = embs
        registro[nombre] = {"alta": datetime.now().isoformat(timespec="seconds"), "por": por, "consentimiento": True}
        _guardar(datos, registro)
    return len(embs), sin_cara


def borrar(nombre):
    """Quita los embeddings y el registro de una persona (derecho de supresión)."""
    datos = _leer_db(estricto=True)
    registro = data_store._load_json("personas_registro.json", {})
    if datos.pop(nombre, None) is None:
        return False
    registro.pop(nombre, None)
    _guardar(datos, registro)
    return True


def _personas():
    """[(nombre, matriz de embeddings normalizados)], recargado si cambia el archivo."""
    ruta = _ruta_db()
    mtime = os.path.getmtime(ruta) if os.path.exists(ruta) else None
    if mtime != _cache["mtime"]:
        datos = _leer_db() if mtime else {}
        _cache["personas"] = [(n, np.array(embs, dtype=np.float32)) for n, embs in datos.items() if embs]
        _cache["mtime"] = mtime
    return _cache["personas"]


def disponible():
    return _sface is not None and face_detector._yunet is not None


def embedding(img):
    """Embedding normalizado de la cara más grande de img, o None."""
    if _sface is None:
        return None
    caras = face_detector.yunet_rows(img)
    if caras is None or len(caras) == 0:
        return None
    cara = max(caras, key=lambda f: f[2] * f[3])
    feat = _feature(_sface.alignCrop(img, cara)).flatten()
    return feat / np.linalg.norm(feat)


def identificar(img, bbox, personas=None):
    """Nombre de la persona cuya cabeza está en bbox [x, y, w, h], "Desconocido",
    o None si no se ve la cara. personas: [(nombre, embeddings)] contra las que comparar;
    por defecto, las dadas de alta."""
    if personas is None:
        personas = _personas()
    if not personas:
        return None
    x, y, w, h = bbox
    H, W = img.shape[:2]
    # La caja de la cabeza puede cortar el mentón: se agranda un poco hacia abajo y a los lados
    x0, y0 = max(0, int(x - 0.2 * w)), max(0, int(y - 0.1 * h))
    x1, y1 = min(W, int(x + 1.2 * w)), min(H, int(y + 1.4 * h))
    emb = embedding(img[y0:y1, x0:x1])
    if emb is None:
        return None
    mejor, nombre = max((float((embs @ emb).max()), n) for n, embs in personas)
    return nombre if mejor >= UMBRAL else "Desconocido"


def enrolar(carpeta=os.path.join(BASE, "personas")):
    """Lee personas/<nombre>/*.jpg|png y suma esas personas a data/personas.json
    (no borra a las registradas desde el panel)."""
    datos, sin_cara = _leer_db(estricto=True), []
    for dir_persona in sorted(glob.glob(os.path.join(carpeta, "*"))):
        if not os.path.isdir(dir_persona):
            continue
        nombre = os.path.basename(dir_persona)
        embs = []
        for ruta in sorted(glob.glob(os.path.join(dir_persona, "*"))):
            img = cv2.imread(ruta)
            if img is None:
                continue
            s = min(1.0, 1280 / max(img.shape[:2]))
            emb = embedding(cv2.resize(img, None, fx=s, fy=s) if s < 1 else img)
            if emb is None:
                sin_cara.append(os.path.relpath(ruta, carpeta))
            else:
                embs.append(emb.tolist())
        datos[nombre] = embs
        print(f"  {nombre}: {len(embs)} fotos útiles")
    _escribir_db(datos)
    if sin_cara:
        print("  Sin cara detectada (no se usaron):", ", ".join(sin_cara))
    return datos


if __name__ == "__main__" and sys.argv[1:] == ["enrolar"]:
    enrolar()
