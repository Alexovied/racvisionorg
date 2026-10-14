"""Chaleco reflectante: se busca color de alta visibilidad en el torso.

Los chalecos de seguridad usan colores fluorescentes normalizados (amarillo
lima o naranja), que casi no aparecen en ropa común ni en fondos. Debajo de
cada cabeza que encuentra el detector de cascos se mide qué parte del torso
tiene ese color.

Medido con fotos del proyecto: con chaleco 9-49 % del torso (9 % de costado),
sin chaleco 0 %, y 0-1 % en 40 fotos del dataset sin chalecos. El umbral 4 %
queda en el medio.

shortcut: regla de color, no un modelo entrenado. Una remera amarilla
fluorescente cuenta como chaleco; si eso pasa en la planta, entrenar YOLO con
fotos de chalecos (mismo notebook que el de cascos).
"""
import cv2

UMBRAL = 0.04


def _torso(img, bbox):
    """Región del torso debajo de la cabeza, o None si casi no entra en la imagen."""
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x0, x1 = max(0, int(x - 0.6 * w)), min(W, int(x + 1.6 * w))
    y0, y1 = min(H, int(y + 1.1 * h)), min(H, int(y + 3.6 * h))
    # Cámara muy cerca (solo se ve la cara): no se puede juzgar el chaleco
    if y1 - y0 < 1.2 * h or x1 - x0 < w:
        return None
    return img[y0:y1, x0:x1]


def proporcion_alta_visibilidad(region):
    hh, ss, vv = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2HSV))
    amarillo = (hh >= 25) & (hh <= 45) & (ss >= 110) & (vv >= 140)
    naranja = (hh >= 4) & (hh <= 18) & (ss >= 150) & (vv >= 150)
    return float((amarillo | naranja).mean())


def tiene_chaleco(img, bbox):
    """True / False, o None si el torso no está a la vista."""
    torso = _torso(img, bbox)
    if torso is None:
        return None
    return proporcion_alta_visibilidad(torso) >= UMBRAL
