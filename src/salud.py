"""Salud de las cámaras y del nodo: un punto ciego es tan peligroso como una infracción.

- lente(img):  "tapada" (negra o pareja: tela, cinta, polvo muy denso), "desenfocada" (sin bordes:
               lente sucia, empañada o movida) o None si se ve bien.
- Vigia:       decide cuándo avisar: el problema tiene que durar SEGUNDOS seguidos y avisa una sola
               vez hasta que la cámara vuelve a estar bien (una nube o una mano no alcanzan).
- nodo():      CPU, memoria y temperatura del equipo que corre el motor.
"""
import cv2

try:
    import psutil
except ImportError:  # sin psutil el panel muestra solo lo que da el sistema operativo
    psutil = None

OSCURO = 18           # brillo medio (0-255) por debajo del cual la imagen está "apagada"
PAREJO = 6            # desvío de brillo: una imagen casi de un solo tono está tapada
NITIDEZ = 15          # varianza del laplaciano a 320 px: menos es una imagen sin bordes
SEGUNDOS = 30


def lente(img):
    gris = cv2.cvtColor(cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    media, desvio = cv2.meanStdDev(gris)
    # Tapada: imagen pareja (tela, cinta). Oscura sola no alcanza: una escena de noche tiene poca luz
    # pero conserva contraste; negra y casi sin contraste, sí es una lente tapada
    if desvio[0][0] < PAREJO or (media[0][0] < OSCURO and desvio[0][0] < 2 * PAREJO):
        return "tapada"
    if cv2.Laplacian(gris, cv2.CV_64F).var() < NITIDEZ:
        return "desenfocada"
    return None


class Vigia:
    """Problema → aviso, con tiempo de confirmación y una sola vez por episodio."""

    def __init__(self, segundos=SEGUNDOS):
        self.segundos = segundos
        self.desde = None
        self.avisado = False

    def ver(self, problema, ahora):
        """Devuelve el problema cuando hay que avisar, si no None."""
        if not problema:
            self.desde, self.avisado = None, False
            return None
        if self.desde is None:
            self.desde = ahora
        if not self.avisado and ahora - self.desde >= self.segundos:
            self.avisado = True
            return problema
        return None


def nodo():
    """{cpu, memoria, temperatura} en porcentaje / °C; None en lo que el equipo no informa."""
    if psutil is None:
        return {"cpu": None, "memoria": None, "temperatura": None}
    temp = None
    try:
        sensores = psutil.sensors_temperatures() or {}
        lecturas = [t.current for grupo in sensores.values() for t in grupo if t.current]
        temp = round(max(lecturas), 1) if lecturas else None
    except (AttributeError, OSError):   # Windows no tiene sensors_temperatures
        pass
    return {"cpu": psutil.cpu_percent(interval=None), "memoria": psutil.virtual_memory().percent,
            "temperatura": temp}
