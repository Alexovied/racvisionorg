"""El cuerpo humano visto por el detector de pose (17 puntos COCO por persona, ver pose.py).

Dos usos:

1. Qué cabezas son de una persona. El detector de cascos mira solo la cabeza, así que un
   casco apoyado en una mesa le parece "una cabeza con casco". Acá se acepta una cabeza
   solo si el esqueleto la respalda: puntos de la cara (nariz, ojos, orejas) adentro de la
   caja y hombros debajo, o de espaldas, los dos hombros justo debajo de ella.

2. Cuánto está girada la cámara. Los modelos se entrenaron con fotos derechas y con 10-20°
   de giro ya pierden cabezas o confunden casco con pelo. La gente parada da la vertical:
   el eje del torso (caderas → hombros) o, si no se ven las caderas, la línea de los hombros.
   Con eso se endereza la imagen antes de buscar cascos y chalecos.
"""
import math
from collections import deque

import cv2
import numpy as np

CARA = (0, 1, 2, 3, 4)            # nariz, ojos, orejas
HOMBROS, CADERAS = (5, 6), (11, 12)
CONF = 0.5                        # mismo criterio que pose.CONF_PUNTO
CONF_HOMBRO = 0.35                # el hombro cortado por el borde o tapado por el brazo sale más bajo

INCLINACION_MIN = 10              # grados: por debajo los modelos andan igual y la medida tiene ese error
INCLINACION_MAX = 45              # más que esto es alguien agachado o caído, no la cámara


def _pts(kp, idx, conf=CONF):
    return [(float(kp[i][0]), float(kp[i][1])) for i in idx if kp[i][2] >= conf]


def es_persona(cabeza, cuerpos, alto):
    """True si la caja de cabeza [x, y, w, h] pertenece a algún esqueleto de `cuerpos`
    [(caja, puntos)], todo en una imagen derecha de `alto` píxeles."""
    x, y, w, h = cabeza
    s = max(w, h)
    cx, cy = x + w / 2, y + h / 2
    m = 0.3 * s
    for _, kp in cuerpos:
        cara = [p for p in _pts(kp, CARA) if x - m <= p[0] <= x + w + m and y - m <= p[1] <= y + h + m]
        # Hombros de esta cabeza: debajo de ella (no al costado ni arriba) y cerca
        hombros = [p for p in _pts(kp, HOMBROS, CONF_HOMBRO)
                   if cy - 0.2 * s <= p[1] <= cy + 2.5 * s and abs(p[0] - cx) <= 2.5 * s]
        if cara and hombros:
            return True
        # Muy cerca de la cámara: la cara se ve y los hombros quedan fuera de la imagen o cortados
        if (len(cara) >= 3 and y + 1.6 * h >= alto) or (cara and h >= 0.4 * alto):
            return True
        # De espaldas (el casco tapa las orejas): la cabeza entre los dos hombros y apenas arriba
        if len(hombros) == 2:
            (x1, y1), (x2, y2) = hombros
            if min(x1, x2) - 0.5 * s <= cx <= max(x1, x2) + 0.5 * s and 0 < min(y1, y2) - cy <= 2 * s:
                return True
    return False


# ── Inclinación de la cámara ─────────────────────────────────────────────────

def _angulo_linea(p, q):
    """Ángulo de la recta p-q respecto de la horizontal, en (-90, 90]."""
    a = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
    while a > 90:
        a -= 180
    while a <= -90:
        a += 180
    return a


def inclinacion(cuerpos):
    """Giro de la imagen en grados (+ = horario) según la gente a la vista, o None si nadie
    permite medirlo. Por persona, la mediana de las líneas que se vean (eje del torso, línea
    de hombros, de ojos y de orejas): cada una sola se equivoca seguido (cuerpo de tres cuartos,
    cabeza ladeada). Entre personas, otra mediana: una sola agachada no mueve el resultado.
    Medido con 16 fotos del proyecto giradas de -30° a 30°: error típico 6°."""
    medidas = []
    for _, kp in cuerpos:
        lineas = []
        hombros, caderas = _pts(kp, HOMBROS), _pts(kp, CADERAS)
        if len(hombros) == 2 and len(caderas) == 2:
            hx, hy = (hombros[0][0] + hombros[1][0]) / 2, (hombros[0][1] + hombros[1][1]) / 2
            cx, cy = (caderas[0][0] + caderas[1][0]) / 2, (caderas[0][1] + caderas[1][1]) / 2
            lineas.append(math.degrees(math.atan2(hx - cx, cy - hy)))   # 0° = torso vertical
        # Líneas que en una persona derecha son horizontales; de perfil quedan muy cortas y no sirven
        for par, minimo in ((HOMBROS, 10), ((1, 2), 5), ((3, 4), 5)):
            pts = _pts(kp, par)
            if len(pts) == 2 and abs(pts[0][0] - pts[1][0]) > minimo:
                lineas.append(_angulo_linea(*pts))
        if lineas:
            a = float(np.median(lineas))
            if abs(a) <= INCLINACION_MAX:
                medidas.append(a)
    return float(np.median(medidas)) if medidas else None


class Nivel:
    """Inclinación de una cámara fija: mediana de lo medido en los últimos frames con gente.
    Si un frame no trae medida (nadie a la vista) se sigue usando la anterior."""

    def __init__(self, n=30):
        self.medidas = deque(maxlen=n)

    def estable(self):
        """Hay medidas suficientes para confiar en el giro aunque cada frame varíe."""
        return len(self.medidas) >= 10

    def actualizar(self, angulo):
        if angulo is not None:
            self.medidas.append(angulo)
        if len(self.medidas) < 3:
            return angulo
        return float(np.median(self.medidas))


def enderezar(img, angulo):
    """Gira la imagen `angulo` grados en sentido antihorario (deshace un giro horario) sin
    recortar las esquinas. Devuelve (imagen, M, M_inversa): M lleva puntos de la original a
    la derecha y M_inversa al revés."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    derecha = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    return derecha, M, cv2.invertAffineTransform(M)


def mover(punto, M):
    return (M[0, 0] * punto[0] + M[0, 1] * punto[1] + M[0, 2],
            M[1, 0] * punto[0] + M[1, 1] * punto[1] + M[1, 2])


def mover_cuerpo(cuerpo, M):
    """(caja, puntos) de la imagen original llevados a la imagen enderezada."""
    caja, kp = cuerpo
    kp = np.array(kp, dtype=np.float32, copy=True)
    kp[:, :2] = kp[:, :2] @ M[:, :2].T + M[:, 2]
    x, y, w, h = caja
    esquinas = np.array([mover(p, M) for p in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))])
    x0, y0 = esquinas.min(axis=0)
    x1, y1 = esquinas.max(axis=0)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], kp


def _recortar(x, y, w, h, forma):
    alto, ancho = forma[:2]
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(ancho, int(round(x + w))), min(alto, int(round(y + h)))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def volver_cabeza(caja, M_inv, forma):
    """Caja de la imagen enderezada → original: mismo tamaño, centrada donde cae su centro."""
    x, y, w, h = caja
    cx, cy = mover((x + w / 2, y + h / 2), M_inv)
    return _recortar(cx - w / 2, cy - h / 2, w, h, forma)


def volver_persona(caja, M_inv, forma):
    """Caja de persona enderezada → original, apoyada en el mismo punto del piso (las zonas de
    exclusión miran el centro del borde de abajo)."""
    x, y, w, h = caja
    px, py = mover((x + w / 2, y + h), M_inv)
    return _recortar(px - w / 2, py - h, w, h, forma)
