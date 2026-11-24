"""Revisión manual de alertas para reentrenar el detector de casco y chaleco.

Cada alerta de cámara guarda, además de la foto con los recuadros (la que se ve en Auditoría),
una copia LIMPIA del cuadro: sin dibujos encima, que es lo que sirve para entrenar. Desde el
monitor también se puede guardar un cuadro a mano ("Guardar para revisar"): así entran los
errores que no generan alerta, como una cabeza sin casco que el modelo vio "con casco".

En /admin/revision un supervisor ve cada foto con las cajas que encontró el modelo, corrige
la etiqueta de cada una (con / sin casco, con / sin chaleco), borra las que no son una persona
y dibuja las que faltan. Las alertas además quedan marcadas como reales o falsas
(data_store.set_feedback), como desde Auditoría.

exportar_zip() arma el dataset en formato YOLO (las mismas clases que
entrenamiento/detector_casco_colab.ipynb: 0 = casco, 1 = sin_casco) para sumarlo al
reentrenamiento en Colab.

Las cajas del modelo se calculan recién cuando alguien abre la foto (detectar()): una alerta no
paga una pasada más de la red, y la propuesta sale del modelo que esté instalado en ese momento.
"""
import csv
import io
import os
import uuid
import zipfile
from datetime import datetime

import cv2
import numpy as np

import data_store

ARCHIVO = "muestras.json"
MAXIMO = 2000            # fotos guardadas; al pasarse se borran primero las viejas sin revisar
CALIDAD = 90             # WebP: alta, porque son para entrenar
MAX_CAJAS = 50
ETIQUETAS = ("con", "sin", None)
TIPOS = ("sin_casco", "sin_chaleco", "zona_prohibida", "hombre_caido")   # alertas de cámara
CLASES_YOLO = ("casco", "sin_casco")


def _carpeta():
    return os.path.join(data_store.DATA_DIR, "muestras")


def ruta_foto(muestra_id):
    return os.path.join(_carpeta(), f"{muestra_id}.webp")


def _id_valido(muestra_id):
    try:
        return str(uuid.UUID(muestra_id)) == muestra_id
    except (ValueError, TypeError, AttributeError):
        return False


def cargar():
    return data_store._load_json(ARCHIVO, [])


def _guardar_lista(muestras):
    data_store._save_json(ARCHIVO, muestras)


def obtener(muestra_id):
    if not _id_valido(muestra_id):
        return None
    return next((m for m in cargar() if m["id"] == muestra_id), None)


def de_evento(event_id):
    return next((m for m in cargar() if m.get("event_id") == event_id), None)


def guardar(foto, camera, origen="alerta", tipo=None, event_id=None):
    """Guarda una foto limpia para revisar. foto: bytes de imagen (JPEG/PNG/WebP) o matriz de
    OpenCV. Se vuelve a codificar siempre (nunca se guardan bytes tal cual llegan).
    Devuelve la muestra, o None si la foto no es una imagen."""
    img = foto
    if isinstance(foto, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(foto, np.uint8), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "ndim", 0) != 3:
        return None
    ok, webp = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, CALIDAD])
    if not ok:
        return None
    alto, ancho = img.shape[:2]
    muestra = {
        "id": str(uuid.uuid4()),
        "origen": origen,                 # "alerta" o "captura"
        "tipo": tipo,                     # tipo de alerta, o None en una captura
        "event_id": event_id,
        "camera": camera,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ancho": ancho, "alto": alto,
        "detecciones": None,              # lo que vio el modelo (se calcula al abrirla)
        "cajas": None,                    # lo que corrigió una persona; None = sin revisar
        "veredicto": None,                # "real" / "falsa" (solo alertas)
    }
    with data_store._lock:
        os.makedirs(_carpeta(), exist_ok=True)
        with open(ruta_foto(muestra["id"]), "wb") as f:
            f.write(webp.tobytes())
        muestras = [muestra] + cargar()
        if len(muestras) > MAXIMO:
            # Se descartan primero las más viejas sin revisar: las revisadas son el dataset
            sobran = len(muestras) - MAXIMO
            orden = sorted(range(len(muestras)),
                           key=lambda i: (muestras[i]["cajas"] is not None, -i))   # la lista va de nueva a vieja
            fuera = {muestras[i]["id"] for i in orden[:sobran]}
            for mid in fuera:
                _borrar_foto(mid)
            muestras = [m for m in muestras if m["id"] not in fuera]
        _guardar_lista(muestras)
    return muestra


def _borrar_foto(muestra_id):
    try:
        os.remove(ruta_foto(muestra_id))
    except FileNotFoundError:
        pass


def borrar(muestra_id):
    with data_store._lock:
        muestras = cargar()
        quedan = [m for m in muestras if m["id"] != muestra_id]
        if len(quedan) == len(muestras):
            return False
        _guardar_lista(quedan)
        _borrar_foto(muestra_id)
    return True


def _actualizar(muestra_id, **campos):
    with data_store._lock:
        muestras = cargar()
        for m in muestras:
            if m["id"] == muestra_id:
                m.update(campos)
                _guardar_lista(muestras)
                return m
    return None


def _norm(caja, ancho, alto):
    x, y, w, h = (float(v) for v in caja)
    return [round(max(0.0, x / ancho), 4), round(max(0.0, y / alto), 4),
            round(min(w / ancho, 1.0), 4), round(min(h / alto, 1.0), 4)]


def detectar(muestra):
    """Cajas que propone el modelo para una muestra (y las deja guardadas). Cabezas del detector
    de casco, con el chaleco de cada una si está el modelo de chalecos. Coordenadas de 0 a 1."""
    if muestra.get("detecciones") is not None:
        return muestra["detecciones"]
    img = cv2.imread(ruta_foto(muestra["id"]), cv2.IMREAD_COLOR)
    if img is None:
        return []
    import epp
    import helmet_detector
    alto, ancho = img.shape[:2]
    cabezas = helmet_detector.detectar(img) if helmet_detector.disponible() else []
    chalecos = epp.chalecos_por_cabeza(img, [c["bbox"] for c in cabezas]) if cabezas else []
    dets = []
    for i, c in enumerate(cabezas):
        chaleco = chalecos[i] if i < len(chalecos) else None
        dets.append({
            "bbox": _norm(c["bbox"], ancho, alto),
            "casco": "con" if c["p_casco"] >= c["p_sin"] else "sin",
            "confianza": round(100 * float(max(c["p_casco"], c["p_sin"])), 1),
            "chaleco": None if chaleco is None else "con" if chaleco else "sin",
        })
    _actualizar(muestra["id"], detecciones=dets)
    muestra["detecciones"] = dets
    return dets


def validar_cajas(cajas):
    """Lista de cajas limpia, o None si el formato no es válido.
    Cada caja: {"bbox": [x, y, w, h] de 0 a 1, "casco": "con"/"sin"/None, "chaleco": igual}."""
    if not isinstance(cajas, list) or len(cajas) > MAX_CAJAS:
        return None
    limpias = []
    for c in cajas:
        if not isinstance(c, dict):
            return None
        b = c.get("bbox")
        if (not isinstance(b, list) or len(b) != 4
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)):
            return None
        x, y, w, h = (float(v) for v in b)
        if not (0 <= x < 1 and 0 <= y < 1 and 0.003 <= w <= 1 and 0.003 <= h <= 1):
            return None
        w, h = min(w, 1 - x), min(h, 1 - y)   # que no se salga de la foto
        casco, chaleco = c.get("casco"), c.get("chaleco")
        if casco not in ETIQUETAS or chaleco not in ETIQUETAS:
            return None
        limpias.append({"bbox": [round(x, 4), round(y, 4), round(w, 4), round(h, 4)],
                        "casco": casco, "chaleco": chaleco})
    return limpias


def revisar(muestra_id, cajas, veredicto, usuario):
    """Guarda las etiquetas corregidas. veredicto: "real"/"falsa" en alertas (marca también la
    alerta en Auditoría), None en capturas."""
    muestra = _actualizar(muestra_id, cajas=cajas, veredicto=veredicto, revisado_por=usuario,
                          revisado_en=datetime.now().isoformat(timespec="seconds"))
    if muestra and muestra.get("event_id") and veredicto in ("real", "falsa"):
        data_store.set_feedback(muestra["event_id"], veredicto, usuario)
    return muestra


def reabrir(muestra_id):
    """Deshace la revisión: vuelve a Sin revisar con las cajas del modelo. En una alerta saca
    también la marca de real / falsa de Auditoría."""
    muestra = _actualizar(muestra_id, cajas=None, veredicto=None, revisado_por=None, revisado_en=None)
    if muestra and muestra.get("event_id"):
        evento = data_store.get_event(muestra["event_id"])
        if evento and "feedback" in evento:
            with data_store._lock:
                eventos = data_store.load_events()
                for e in eventos:
                    if e.get("id") == muestra["event_id"]:
                        e.pop("feedback", None)
                data_store._save_json("events.json", eventos)
    return muestra


def resumen(muestras=None):
    muestras = cargar() if muestras is None else muestras
    revisadas = [m for m in muestras if m["cajas"] is not None]
    cajas = [c for m in revisadas for c in m["cajas"]]
    return {
        "total": len(muestras),
        "pendientes": len(muestras) - len(revisadas),
        "revisadas": len(revisadas),
        "falsas": sum(1 for m in revisadas if m.get("veredicto") == "falsa"),
        "con_casco": sum(1 for c in cajas if c["casco"] == "con"),
        "sin_casco": sum(1 for c in cajas if c["casco"] == "sin"),
        "con_chaleco": sum(1 for c in cajas if c["chaleco"] == "con"),
        "sin_chaleco": sum(1 for c in cajas if c["chaleco"] == "sin"),
    }


def lineas_yolo(cajas):
    """Líneas YOLO (clase cx cy w h) de las cabezas con casco conocido."""
    lineas = []
    for c in cajas:
        if c["casco"] not in ("con", "sin"):
            continue
        x, y, w, h = c["bbox"]
        clase = 0 if c["casco"] == "con" else 1
        lineas.append(f"{clase} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
    return lineas


LEEME = """Dataset de RAC Vision: alertas y capturas revisadas a mano
Exportado el {fecha} ({n} fotos).

images/ + labels/ + data.yaml
    Formato YOLO, las mismas clases que entrenamiento/detector_casco_colab.ipynb:
    0 = casco (cabeza con casco), 1 = sin_casco (cabeza sin casco).
    Una foto con el .txt vacío no tiene ninguna cabeza: le enseña al modelo que ahí no hay
    nada (por ejemplo, un casco sobre una mesa que el modelo marcaba como persona).
    Las cabezas marcadas "no se ve" no van en el .txt.

con_casco/ y sin_casco/
    Las fotos con una sola cabeza, separadas por carpeta: es el formato de fotos_rac.zip
    que piden los notebooks de entrenamiento (paso "Subir tus fotos").

etiquetas.csv
    Todas las cajas con casco y chaleco (con / sin / vacío = no se ve), en coordenadas
    de 0 a 1 (x, y = esquina de arriba a la izquierda), y si la alerta fue real o falsa.

Para sumarlo al reentrenamiento en Colab (después de convertir el dataset público):
    import zipfile, shutil; from pathlib import Path
    zipfile.ZipFile("rac_revisadas.zip").extractall("/content/rac_rev")
    for img in Path("/content/rac_rev/images").glob("*.jpg"):
        for k in range(3):   # que pesen frente a las miles de fotos públicas
            shutil.copy(img, DST / "images" / "train" / f"rev_{{img.stem}}_{{k}}.jpg")
            shutil.copy(Path("/content/rac_rev/labels") / (img.stem + ".txt"),
                        DST / "labels" / "train" / f"rev_{{img.stem}}_{{k}}.txt")
"""


def exportar_zip(muestras=None):
    """(bytes del zip, cantidad de fotos) con las muestras revisadas."""
    muestras = cargar() if muestras is None else muestras
    buf = io.BytesIO()
    filas = []
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for m in muestras:
            if m.get("cajas") is None:
                continue
            img = cv2.imread(ruta_foto(m["id"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            nombre = f"rac_{m['id'][:8]}"
            z.writestr(f"images/{nombre}.jpg", jpg.tobytes())
            z.writestr(f"labels/{nombre}.txt", "\n".join(lineas_yolo(m["cajas"])))
            con_casco = [c for c in m["cajas"] if c["casco"] in ("con", "sin")]
            if len(m["cajas"]) == 1 and con_casco:
                carpeta = "con_casco" if con_casco[0]["casco"] == "con" else "sin_casco"
                z.writestr(f"{carpeta}/{nombre}.jpg", jpg.tobytes())
            for c in m["cajas"] or [{}]:
                filas.append([f"{nombre}.jpg", m.get("origen"), m.get("tipo") or "", m.get("veredicto") or "",
                              m.get("camera") or "", m.get("timestamp") or "",
                              *(c.get("bbox") or ["", "", "", ""]), c.get("casco") or "", c.get("chaleco") or ""])
            n += 1
        z.writestr("data.yaml", "path: .\ntrain: images\nval: images\nnames:\n  0: casco\n  1: sin_casco\n")
        tabla = io.StringIO()
        w = csv.writer(tabla)
        w.writerow(["imagen", "origen", "alerta", "veredicto", "camara", "fecha", "x", "y", "ancho", "alto",
                    "casco", "chaleco"])
        w.writerows(filas)
        z.writestr("etiquetas.csv", tabla.getvalue())
        z.writestr("LEEME.txt", LEEME.format(fecha=datetime.now().strftime("%d/%m/%Y %H:%M"), n=n))
    return buf.getvalue(), n
