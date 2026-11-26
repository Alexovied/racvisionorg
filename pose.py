"""Pose de las personas (YOLO11n-pose exportado a ONNX): 17 puntos del cuerpo por persona.

El motor de cámaras lo usa para el hombre caído: mide la inclinación del torso
(hombros → caderas) en vez de mirar solo la forma de la caja. Y cuerpo.py lo usa para
descartar cabezas sin cuerpo (un casco sobre la mesa) y enderezar una cámara girada.
Si detector_pose.onnx no está, `disponible()` devuelve False y motor.py sigue como antes:
regla de la caja, todas las cabezas cuentan y no se corrige el giro.
"""
import math
import os
import threading

import cv2

import helmet_detector

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_pose.onnx")
# Valores elegidos probando 8.100 combinaciones con clips de Pexels (demo-persona-caida/herramientas/
# buscar_reglas.py): con estos y YOLO11n-pose, alerta en 6/6 caídas de entrenamiento y 7/8 de validación,
# sin falsas alarmas en 11 clips de gente parada, caminando, agachada o arrodillada.
CONF_PERSONA = 0.25
CONF_PUNTO = 0.3          # por debajo, el punto no se ve bien y no se usa
ANGULO_CAIDO = 40         # grados del torso respecto de la vertical
# Agachado o sentado inclinado hacia adelante: el torso se ve "acostado" pero las piernas bajan
# casi verticales desde la cadera. Con GMDCSA24 (160 videos, 4 personas) esto bajó las falsas alarmas
# de 11 a 1 de 63 clips sin caída. El largo mínimo evita descartar a alguien hecho un bollo en el piso.
PIERNAS_DERECHAS = 25     # grados de cadera → tobillos (o rodillas) respecto de la vertical
LARGO_PIERNAS = 0.9       # las piernas tienen que medir al menos 0,9 veces el torso
# Alguien tirado en el piso a lo largo de la imagen se le escapa al modelo de pose (lo entrenaron con
# gente parada): se vuelve a buscar con la imagen girada 90° a cada lado. Solo cuenta si la detección es
# muy segura y la caja es bien acostada. En GMDCSA24 pasó de 55 a 62 de 79 caídas vistas sin falsas nuevas.
CONF_GIRADA = 0.7
ANCHO_GIRADA = 1.6        # ancho de la caja / alto
RODILLAS, TOBILLOS = (13, 14), (15, 16)
NARIZ, HOMBROS, CADERAS = 0, (5, 6), (11, 12)   # índices COCO

_net, _lock = None, threading.Lock()
if os.path.exists(MODEL_PATH):
    try:
        _net, motor = helmet_detector.cargar_modelo(MODEL_PATH)
        print(f"[+] Detector de pose: YOLO (detector_pose.onnx) con {motor}")
    except cv2.error as e:
        print(f"[!] No se pudo cargar detector_pose.onnx ({e}); el hombre caído usa la forma de la caja")


def disponible():
    return _net is not None


def detectar(img):
    """[(bbox [x, y, w, h], puntos (17, 3): x, y, confianza)] de cada persona."""
    if _net is None:
        return []
    return [(bbox, puntos) for bbox, _, puntos in
            helmet_detector.correr_yolo(_net, _lock, img, 1, CONF_PERSONA, puntos=True)]


def detectar_girados(img):
    """Como detectar(), pero con la imagen girada 90° a cada lado (puntos y cajas vueltos a la imagen
    original). Solo personas con confianza ≥ CONF_GIRADA y caja ≥ ANCHO_GIRADA veces más ancha que alta."""
    if _net is None:
        return []
    alto, ancho = img.shape[:2]
    res = []
    for giro in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        for (x, y, w, h), _, kp in helmet_detector.correr_yolo(_net, _lock, cv2.rotate(img, giro), 1,
                                                               CONF_GIRADA, puntos=True):
            kp = kp.copy()
            if giro == cv2.ROTATE_90_CLOCKWISE:      # (x', y') girado → (y', alto-1-x') original
                caja = [y, alto - x - w, h, w]
                kp[:, 0], kp[:, 1] = kp[:, 1].copy(), alto - 1 - kp[:, 0]
            else:                                    # (x', y') → (ancho-1-y', x')
                caja = [ancho - y - h, x, h, w]
                kp[:, 0], kp[:, 1] = ancho - 1 - kp[:, 1], kp[:, 0].copy()
            if caja[2] >= ANCHO_GIRADA * caja[3]:
                res.append((caja, kp))
    return res


def _angulo(arriba, abajo, inclinacion):
    """Grados entre el vector abajo → arriba y la vertical (0 = derecho), corrigiendo el giro de la cámara."""
    angulo = math.degrees(math.atan2(arriba[0] - abajo[0], abajo[1] - arriba[1])) - inclinacion
    return abs((angulo + 180) % 360 - 180)


def _piernas_derechas(puntos, caderas, torso, inclinacion):
    """True si los tobillos (o, si no se ven, las rodillas) están casi en vertical debajo de la cadera
    y las piernas son largas: agachado o sentado inclinado, no tirado."""
    for pie in (TOBILLOS, RODILLAS):
        if min(puntos[i][2] for i in pie) >= CONF_PUNTO:
            px, py = (puntos[pie[0]][0] + puntos[pie[1]][0]) / 2, (puntos[pie[0]][1] + puntos[pie[1]][1]) / 2
            largo = math.hypot(px - caderas[0], py - caderas[1])
            return _angulo(caderas, (px, py), inclinacion) < PIERNAS_DERECHAS and largo > LARGO_PIERNAS * torso
    return False


def torso_caido(puntos, inclinacion=0.0):
    """True si el torso está acostado o cabeza abajo; False si está derecho;
    None si no se ven bien los hombros y las caderas (ni la cabeza) (muy cerca de la cámara, tapado).
    inclinacion: grados que está girada la cámara (horario), ver cuerpo.inclinacion()."""
    if min(puntos[i][2] for i in HOMBROS) < CONF_PUNTO:
        return None
    hx, hy = (puntos[5][0] + puntos[6][0]) / 2, (puntos[5][1] + puntos[6][1]) / 2
    if min(puntos[i][2] for i in CADERAS) >= CONF_PUNTO:
        cx, cy = (puntos[11][0] + puntos[12][0]) / 2, (puntos[11][1] + puntos[12][1]) / 2
    elif puntos[NARIZ][2] >= CONF_PUNTO:
        # Caderas tapadas (frazada, de cerca, cortado por el borde): se mide hombros → cabeza,
        # que en alguien de pie también apunta para arriba.
        cx, cy = hx, hy
        hx, hy = puntos[NARIZ][0], puntos[NARIZ][1]
    else:
        return None
    # Ángulo entre el torso y la vertical: 0° de pie, 90° acostado, más de 90° cabeza abajo.
    # shortcut: alguien tirado a lo largo del eje de la cámara se ve "vertical" en la imagen;
    # para eso hace falta saber la altura de la cámara o una segunda cámara.
    if _angulo((hx, hy), (cx, cy), inclinacion) <= ANGULO_CAIDO:
        return False
    if min(puntos[i][2] for i in CADERAS) >= CONF_PUNTO and \
            _piernas_derechas(puntos, (cx, cy), math.hypot(hx - cx, hy - cy), inclinacion):
        return False
    return True
