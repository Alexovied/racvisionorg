"""Motor de cámaras del servidor.

Un hilo por cada cámara activa de tipo "usb" (enchufada a este equipo) o "ip" (RTSP/HTTP).
Las "relay" están en otra red (la del cliente): un programa allá (herramientas/relay_camara.py)
manda los cuadros a /admin/api/cameras/<id>/cuadro y este hilo los analiza igual que los demás.
Cada hilo lee el video, corre las mismas detecciones que /predict y decide las alertas acá,
en el servidor: sigue vigilando aunque nadie tenga el panel abierto. Las cámaras "webcam"
siguen como antes (el navegador manda los frames a /predict y decide con static/epp.js).

Además del casco y el chaleco mira dos cosas que necesitan la caja de la persona entera:
  - zonas de exclusión: polígonos por cámara; alerta si alguien pisa adentro, con o sin EPP.
  - hombre caído: una persona acostada durante `caida_segundos` (Configuración).
"""
import os
import threading
import time
import traceback
import unicodedata
from datetime import datetime
from collections import deque

import cv2
import numpy as np

import cuerpo
import data_store
import epp
import face_id
import helmet_detector
import pose
import salud

# Mismas reglas que static/epp.js: ventana de 8 frames, alerta con 60 % "falta", OK con 70 % y ventana llena
VENTANA, TH_FALTA, TH_OK, VACIOS_RESET = 8, 0.60, 0.70, 15
REARME_S = 30          # cámara sin nadie este tiempo: una alerta de casco / chaleco puede volver a avisar
TIMEOUT_IP_MS = 8000   # abrir / leer una cámara IP: más que esto es "sin señal"
ANCHO_MAX = 960        # los frames más anchos se achican al leerlos
ANALISIS_RELAY_S = 0.5 # relay: el video llega a ~8 cuadros/s para verse fluido, pero se analizan 2 por segundo
EVENTOS = {"casco": "sin_casco", "chaleco": "sin_chaleco", "zona": "zona_prohibida", "caida": "hombre_caido"}
TITULOS = {"casco": "SIN CASCO", "chaleco": "SIN CHALECO", "zona": "ZONA PROHIBIDA", "caida": "HOMBRE CAÍDO"}
ROJO, VERDE, AMARILLO = (68, 68, 239), (129, 185, 16), (0, 200, 255)  # BGR

# admin.py pone acá su registrar_evento (evento + foto + alarma + webhook): este módulo no importa al panel
registrar_evento = None
# ...y su guardar_muestra(evento, jpg): la foto limpia (sin recuadros) para revisar y reentrenar (revision.py)
guardar_muestra = None


class _Pendiente(tuple):
    """Argumentos de registrar_evento, y la foto limpia del cuadro si es una alerta."""
    limpia = None


def analizar(img, modos, con_personas=False, conocidos=None, nivel=None, info=None):
    """Detecciones de un frame: (caras, personas).
    caras: lo que /predict devuelve en "faces". personas: cajas [x, y, w, h] de cuerpo entero,
    solo si con_personas (cuesta una pasada del modelo de chalecos aunque no haya cabezas).
    conocidos: caras contra las que identificar (por defecto, las dadas de alta).
    nivel: cuerpo.Nivel de la cámara (inclinación promediada); sin él se mide en este frame.
    info: si es un dict, se completa con "cuerpos" (esqueletos de pose, en la imagen original)
    e "inclinacion" (grados que se enderezó la imagen), para no volver a correr la pose.

    Con el modelo de pose: solo cuentan las cabezas que tienen un cuerpo (un casco apoyado en
    una mesa no es una persona) y, si la cámara está girada, se endereza la imagen antes de
    buscar cascos y chalecos. Las cajas devueltas son siempre de la imagen original."""
    cuerpos = pose.detectar(img) if pose.disponible() and (modos["casco"] or modos["chaleco"] or con_personas) else None
    angulo = cuerpo.inclinacion(cuerpos) if cuerpos else None
    if nivel is not None:
        angulo = nivel.actualizar(angulo)
    if angulo is None or abs(angulo) < cuerpo.INCLINACION_MIN:
        angulo = 0.0
    if info is not None:
        info.update(cuerpos=cuerpos, inclinacion=angulo)
    if not angulo:
        return _analizar(img, modos, con_personas, conocidos, cuerpos)
    if nivel is None or not nivel.estable():
        # Giro medido en un solo frame: también cambia si la persona ladea la cabeza o se
        # recuesta, y enderezar de más confunde pelo con casco. Se usa solo si la imagen
        # tal cual no encuentra a nadie (el caso de la cámara girada que pierde cabezas).
        caras, personas = _analizar(img, modos, con_personas, conocidos, cuerpos)
        if caras:
            # No se enderezó: ese giro puede ser un compañero agachado, y restárselo al torso
            # de un caído (acostados) lo haría pasar por alguien de pie.
            if info is not None:
                info["inclinacion"] = 0.0
            return caras, personas

    derecha, M, M_inv = cuerpo.enderezar(img, angulo)
    caras, personas = _analizar(derecha, modos, con_personas, conocidos,
                                [cuerpo.mover_cuerpo(c, M) for c in cuerpos or []] if cuerpos is not None else None)
    for c in caras:
        c["bbox"] = cuerpo.volver_cabeza(c["bbox"], M_inv, img.shape)
    return caras, [cuerpo.volver_persona(p, M_inv, img.shape) for p in personas]


def _analizar(img, modos, con_personas, conocidos, cuerpos):
    """analizar() sobre una imagen ya derecha. cuerpos: esqueletos de esa imagen, o None sin
    modelo de pose (entonces no se filtran las cabezas, como antes)."""
    dets = epp.detectar(img) if con_personas else None
    personas = [b for b, sc in dets or [] if sc.argmax() == 0]

    if not modos["casco"] and modos["chaleco"] and epp.disponible():
        # Solo chaleco: el modelo de chalecos encuentra a las personas por su cuenta
        return [{"bbox": p["bbox"], "chaleco": p["chaleco"],
                 "nombre": face_id.identificar(img, p["bbox"], conocidos)}
                for p in epp.detectar_personas(img, dets)], personas

    # Detector de cabezas (YOLO): ve el casco desde cualquier ángulo, aunque
    # la persona esté de espaldas o se tape la cara.
    # Con pose y sin ningún cuerpo a la vista todas las cabezas se descartarían: no se buscan.
    cabezas = [] if cuerpos is not None and not cuerpos else helmet_detector.detectar(img)
    if cuerpos is not None:
        # Solo las cabezas con un cuerpo debajo: un casco sobre la mesa no es una persona
        alto = img.shape[0]
        cabezas = [c for c in cabezas if cuerpo.es_persona(c["bbox"], cuerpos, alto)]
    chalecos = epp.chalecos_por_cabeza(img, [c["bbox"] for c in cabezas], dets) if modos["chaleco"] else []
    caras = []
    for i, c in enumerate(cabezas):
        # None = no se ve la cara; "Desconocido" = cara que no está dada de alta
        cara = {"bbox": c["bbox"], "nombre": face_id.identificar(img, c["bbox"], conocidos)}
        if modos["casco"]:
            cara.update(con_casco=c["p_casco"] * 100, sin_casco=c["p_sin"] * 100)
        if modos["chaleco"]:
            cara["chaleco"] = chalecos[i]
        caras.append(cara)
    return caras, personas


# ── Reglas (port de static/epp.js) ───────────────────────────────────────────

def _evaluar(c, min_conf):
    """Estado de cada elemento en una persona: 'ok', 'falta' o None (no se puede juzgar)."""
    casco = None
    if c.get("con_casco") is not None and max(c["con_casco"], c["sin_casco"]) >= min_conf:
        casco = "ok" if c["con_casco"] > c["sin_casco"] else "falta"
    return {"casco": casco, "chaleco": {True: "ok", False: "falta"}.get(c.get("chaleco"))}


def en_horario(horario, ahora):
    """True si a esa hora se exige EPP. horario: {"desde": "07:00", "hasta": "19:00"}; vacío = siempre.
    Si desde > hasta el turno cruza la medianoche (22:00 a 06:00)."""
    if not horario or horario.get("desde") == horario.get("hasta"):
        return True
    t, d, h = ahora.strftime("%H:%M"), horario["desde"], horario["hasta"]
    return d <= t < h if d < h else t >= d or t < h


def _peor(estados):
    """Voto del frame: la peor situación entre las personas a la vista."""
    return "falta" if "falta" in estados else "ok" if "ok" in estados else None


def _acostada(p, cabezas, ancho, alto):
    """Persona acostada: caja bastante más ancha que alta, entera dentro de la imagen.
    shortcut: mira solo la forma de la caja. No ve a alguien caído hacia la cámara (caja casi
    cuadrada) ni a quien queda cortado por el borde; cuando haga falta más, usar un modelo de
    pose y medir el ángulo del torso."""
    x, y, w, h = p
    if w < 1.3 * h or x < 4 or y < 4 or x + w > ancho - 4 or y + h > alto - 4:
        return False
    # Sentado cerca de la cámara la caja también sale ancha, pero la cabeza ocupa buena parte
    # de ese ancho; en alguien tirado en el piso es una punta de la caja.
    return not any(c[2] > 0.3 * w and x <= c[0] + c[2] / 2 <= x + w and y <= c[1] + c[3] / 2 <= y + h
                   for c in cabezas)


def _torso_caido(puntos, inclinacion):
    """pose.torso_caido, mirando también sin corregir el giro de la cámara: si la única persona
    a la vista está tirada en diagonal, cuerpo.Nivel la toma por una cámara girada y "endereza"
    justo la caída."""
    torso = pose.torso_caido(puntos, inclinacion)
    if torso is False and inclinacion:
        return pose.torso_caido(puntos, 0.0)
    return torso


def acostados(img, personas, caras, modos, cuerpos=None, inclinacion=0.0):
    """Cajas de las personas acostadas. Con modelo de pose decide el torso (acostado o cabeza
    abajo); si no se ven hombros y caderas, o no hay modelo, decide la forma de la caja.
    cuerpos / inclinacion: lo que dejó analizar() en `info`, para no correr la pose otra vez
    y medir el torso respecto de la vertical real aunque la cámara esté girada."""
    alto, ancho = img.shape[:2]
    if pose.disponible():  # sin detector de chalecos `personas` viene vacío: la pose igual ve a la gente
        if cuerpos is None:
            cuerpos = pose.detectar(img)
        candidatos = [(caja, _torso_caido(puntos, inclinacion)) for caja, puntos in cuerpos]
    else:
        candidatos = [(p, None) for p in personas]
    caidos = [caja for caja, torso in candidatos if torso]
    if not caidos and pose.disponible():
        # Tirado a lo largo de la imagen el modelo de pose a veces no lo ve: se mira girado 90°
        caidos = [caja for caja, puntos in pose.detectar_girados(img) if pose.torso_caido(puntos, 0.0)]
    dudosos = [caja for caja, torso in candidatos if torso is None and _acostada(caja, [], ancho, alto)]
    if dudosos:
        # Solo chaleco: las "caras" son cajas de persona, así que las cabezas se buscan aparte
        cabezas = [c["bbox"] for c in (caras if modos["casco"] else helmet_detector.detectar(img))]
        caidos += [caja for caja in dudosos if _acostada(caja, cabezas, ancho, alto)]
    return caidos


class Ventana:
    """Votos recientes de un elemento. votar() devuelve el estado nuevo ('ok' / 'falta')
    cuando cambia, o None si sigue igual o todavía no hay consenso."""

    def __init__(self):
        self.votos, self.estado = deque(maxlen=VENTANA), None
        self.rearmada = False  # en "falta" y la cámara quedó vacía: otro consenso "falta" avisa de nuevo

    def votar(self, v, minimo):
        self.votos.append(v)
        n, faltas = len(self.votos), self.votos.count("falta")
        # A diferencia de epp.js, "falta" exige `minimo` votos (Frames de Confirmación):
        # un solo frame malo sobre la ventana vacía no dispara la alarma.
        if faltas >= minimo and faltas / n >= TH_FALTA:
            decidido = "falta"
        elif n == VENTANA and self.votos.count("ok") / n >= TH_OK:
            decidido = "ok"
        else:
            return None
        if decidido == self.estado and not self.rearmada:
            return None
        self.estado, self.rearmada = decidido, False
        return decidido

    def reset(self):
        self.votos.clear()
        self.estado, self.rearmada = None, False


class Seguimiento:
    """Estado temporal de una cámara: una ventana por elemento y el reloj del hombre caído."""

    def __init__(self):
        self.ventanas = {k: Ventana() for k in ("casco", "chaleco", "zona")}
        self.vacios, self.t_gente = 0, 0.0   # t_gente: último voto de casco o chaleco
        self.acostado, self.t_ultimo = deque(maxlen=VENTANA), None
        self.caido_desde, self.caido = None, False

    def procesar(self, votos, hay_caido, minimo, caida_segundos, ahora):
        """votos: {elemento activo: 'ok' / 'falta' / None}. Devuelve los elementos que pasan a alerta."""
        for k, ventana in self.ventanas.items():
            if k not in votos:
                ventana.reset()  # elemento apagado
        # Cámara vacía: los verdes de EPP dejan de valer; las alertas siguen hasta ver a alguien con EPP
        if votos.get("casco") or votos.get("chaleco"):
            self.vacios, self.t_gente = 0, ahora
        else:
            self.vacios += 1
            if self.vacios >= VACIOS_RESET:
                # Tras REARME_S sin nadie los votos sueltos vencen y la alerta, que sigue a la vista,
                # se rearma: si no, el próximo infractor no avisa hasta que pase alguien con EPP.
                viejo = ahora - self.t_gente >= REARME_S
                for k in ("casco", "chaleco"):
                    ventana = self.ventanas[k]
                    if ventana.estado == "ok":
                        ventana.reset()
                    elif viejo:
                        ventana.votos.clear()
                        ventana.rearmada = ventana.estado == "falta"
        minimo = max(1, min(VENTANA, minimo))
        nuevas = [k for k, v in votos.items() if v and self.ventanas[k].votar(v, minimo) == "falta"]

        # Hombre caído: acostado en la mayoría de los últimos frames, de forma sostenida.
        # Un hueco sin video (cámara trabada) no cuenta como tiempo en el suelo. El hueco tolerado
        # crece con caida_segundos: un relay con microcortes o un servidor con muchas cámaras dejan
        # más de 2 s entre cuadros, y con 2 s fijos la cuenta volvía a cero cada vez.
        if self.t_ultimo is not None and ahora - self.t_ultimo > max(2.0, caida_segundos / 2):
            self.acostado.clear()
            self.caido_desde = None
        self.t_ultimo = ahora
        self.acostado.append(hay_caido)
        if sum(self.acostado) / len(self.acostado) < TH_FALTA:
            self.caido_desde, self.caido = None, False
        elif self.caido_desde is None:
            self.caido_desde = ahora
        if self.caido_desde is not None and not self.caido and ahora - self.caido_desde >= caida_segundos:
            self.caido = True
            nuevas.append("caida")
        return nuevas

    def resumen(self, activos):
        """('alerta' / 'ok' / None, título) con el estado combinado de la cámara."""
        faltan = [k for k in activos if self.ventanas[k].estado == "falta"] + (["caida"] if self.caido else [])
        if faltan:
            return "alerta", " · ".join(TITULOS[k] for k in faltan)
        epp_activos = [k for k in activos if k != "zona"]
        if epp_activos and all(self.ventanas[k].estado == "ok" for k in epp_activos):
            return "ok", "TODOS CON EPP"
        return None, None


# ── Dibujo (la misma imagen sirve de video en vivo y de foto de la alerta) ───

def _ascii(texto):
    """cv2.putText no dibuja tildes ni eñes."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def _recuadro(img, caja, texto, color):
    x, y, w, h = caja
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y - th - 8, 0)
    cv2.rectangle(img, (x, ty), (x + tw + 8, ty + th + 8), color, -1)
    cv2.putText(img, texto, (x + 4, ty + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _etiqueta(c, e):
    partes = [("CON " if e[k] == "ok" else "SIN ") + k.upper() for k in ("casco", "chaleco") if e[k]]
    return _ascii(" - ".join(([c["nombre"]] if c.get("nombre") and partes else []) + partes))


# ── Cámara ───────────────────────────────────────────────────────────────────

def _achicar(img):
    """Cámaras Full HD o más: a 960 de ancho. Analizar, dibujar y mandar al panel 1080p
    satura la CPU, y los recuadros y letras quedan diminutos en la tarjeta."""
    if img.shape[1] <= ANCHO_MAX:
        return img
    return cv2.resize(img, (ANCHO_MAX, round(img.shape[0] * ANCHO_MAX / img.shape[1])), interpolation=cv2.INTER_AREA)


class Camara(threading.Thread):
    def __init__(self, cam, anteriores=()):
        super().__init__(daemon=True, name="camara-" + cam["id"][:8])
        self.cam = cam  # sincronizar() lo reemplaza cuando cambian las zonas
        self.anteriores = anteriores  # hilos que se están cerrando: pueden tener abierto el mismo dispositivo
        self.pendientes = []          # eventos decididos y todavía sin registrar
        self.parar = threading.Event()
        self.seg = Seguimiento()
        self.cfg, self.t_cfg = None, 0.0
        self.alarma, self.t_sin_alerta = None, None   # t_sin_alerta: desde cuándo la alarma no tiene alerta a la vista
        self.jpg, self.fps, self.error = None, 0.0, "Conectando"
        self.crudo = None             # último frame leído de la cámara, sin dibujar
        self.t_recibido = 0.0         # cámaras relay: cuándo llegó el último cuadro
        self.n_recibido = -1          # y su número: el relay manda varios a la vez y pueden llegar desordenados
        self.pintar = None            # dibuja sobre un frame los recuadros del último análisis
        self._vista = None            # último JPEG de vista(): (clave, jpg, frame)
        self.caidos, self.t_pose = [], 0.0
        # Inclinación de esta cámara, promediada en minutos: en segundos la medía la postura de trabajo.
        # n=40: un giro equivocado (o la cámara movida) se corrige en ~200 medidas, no en 600
        self.nivel = cuerpo.Nivel(n=40, cada=10)
        self.estado = {"clase": None, "titulo": "CONECTANDO"}
        # Salud: lente tapada / desenfocada y señal perdida avisan si duran (salud.SEGUNDOS / 60 s)
        self.lente, self.t_lente = None, 0.0
        self.vigia_lente = salud.Vigia()
        self.vigia_senal = salud.Vigia(60)
        self.exigia = True            # si en el cuadro anterior se exigía EPP (horario de la cámara)

    def recibir(self, img, numero=None):
        """Cuadro que mandó el relay de una cámara "relay". Uno más viejo que el último se descarta;
        un número bajo de golpe es que el relay se reinició."""
        if numero is not None:
            if self.n_recibido - 50 < numero <= self.n_recibido:
                return
            self.n_recibido = numero
        self.crudo, self.t_recibido = _achicar(img), time.monotonic()
        self.vigia_senal.ver(None, 0)

    def _esperar_relay(self):
        """Cámara relay: no hay nada que leer, solo marcar "sin señal" si dejan de llegar cuadros."""
        while not self.parar.wait(1):
            if self.crudo is not None and time.monotonic() - self.t_recibido > TIMEOUT_IP_MS / 1000:
                self.crudo, self.jpg, self.fps, self.error = None, None, 0.0, "Sin señal"
            if self.crudo is None:
                self._sin_senal(time.monotonic())

    def _abrir(self):
        if self.cam["type"] == "usb":
            # DirectShow abre las webcams USB en Windows mucho más rápido que el backend por defecto
            cap = cv2.VideoCapture(int(self.cam.get("url") or 0), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        # Cámara IP (RTSP / HTTP): con timeouts, si no un stream colgado deja el último frame congelado
        # como si fuera en vivo y read() no vuelve nunca
        return cv2.VideoCapture(self.cam.get("url", ""), cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_IP_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_IP_MS])

    def _sin_senal(self, ahora):
        """Sin frames: registra 'camara_sin_senal' si dura 60 s (una sola vez por corte)."""
        if self.vigia_senal.ver("camara_sin_senal", ahora):
            try:
                registrar_evento("camara_sin_senal", 100, self.cam["id"])
            except Exception:
                traceback.print_exc()
                self.vigia_senal.avisado = False   # no se pudo guardar: el próximo llamado reintenta

    def run(self):
        for h in self.anteriores:
            h.join()
        # Este hilo solo lee la cámara, a su ritmo: el video del panel se ve fluido y nunca se
        # acumulan frames viejos. El análisis va en otro hilo y toma siempre el frame más nuevo.
        threading.Thread(target=self._analizar, daemon=True, name=self.name + "-analisis").start()
        if self.cam["type"] == "relay":
            return self._esperar_relay()
        espera = 5
        while not self.parar.is_set():
            cap = None
            try:  # un error de OpenCV (driver, URL rara) no debe matar el hilo: se reintenta igual
                cap = self._abrir()
                t_abierta = time.monotonic()
                while not self.parar.is_set():
                    ok, img = cap.read()
                    if not ok:
                        break
                    self.crudo = _achicar(img)
                    # Volvió la señal (y se sostiene 5 s: un stream que da un cuadro y muere no cuenta)
                    if espera != 5 and time.monotonic() - t_abierta >= 5:
                        espera = 5
                        self.vigia_senal.ver(None, 0)   # el próximo corte avisa de nuevo
            except Exception:
                traceback.print_exc()
            finally:
                if cap is not None:
                    cap.release()
            self.crudo, self.jpg, self.fps, self.error = None, None, 0.0, "Sin señal"
            self._sin_senal(time.monotonic())
            # Se vuelve a intentar (cámara desenchufada o red caída), cada vez más espaciado hasta 1 minuto
            self.parar.wait(espera)
            espera = min(60, espera * 2)

    def _analizar(self):
        ultimo, t = None, time.monotonic()
        ultimo_t = t
        while not self.parar.is_set():
            img = self.crudo
            if img is None or img is ultimo:
                self.parar.wait(0.005)
                continue
            ultimo = img
            try:
                self._procesar(img.copy())
                self.error = None
            except Exception:  # un frame que falla no tumba la cámara
                traceback.print_exc()
                self.error = "Error de análisis"
                self.parar.wait(1)
            ahora = time.monotonic()
            self.fps = 0.8 * self.fps + 0.2 / max(ahora - t, 1e-3)
            t = ahora
            if self.cam["type"] == "relay":
                self.parar.wait(max(0.0, ANALISIS_RELAY_S - (time.monotonic() - ultimo_t)))
            ultimo_t = time.monotonic()

    def vista(self):
        """JPEG del frame más nuevo con los recuadros del último análisis, o None sin señal."""
        img, pintar = self.crudo, self.pintar
        if img is None:
            return None
        # El mismo cuadro con los mismos recuadros no se vuelve a codificar: cada video abierto en
        # el panel pide 20 por segundo, y codificar de más le saca CPU al análisis
        clave = (id(img), id(pintar))
        cache = self._vista
        if cache and cache[0] == clave and cache[2] is img:
            return cache[1]
        img2 = img.copy()
        if pintar:
            pintar(img2)
        ok, jpg = cv2.imencode(".jpg", img2, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg = jpg.tobytes() if ok else None
        self._vista = (clave, jpg, img)
        return jpg

    def _procesar(self, img):
        ahora = time.monotonic()
        if self.cfg is None or ahora - self.t_cfg > 2:  # la configuración se relee cada 2 s, no por frame
            self.cfg, self.t_cfg = data_store.load_config(), ahora
        cfg, cam = self.cfg, self.cam
        alto, ancho = img.shape[:2]
        if ahora - self.t_lente >= 2:  # lente tapada o desenfocada: barato, pero no hace falta en cada frame
            self.lente, self.t_lente = salud.lente(img), ahora
            problema = self.vigia_lente.ver(self.lente and "camara_" + self.lente, ahora)
            if problema:
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.pendientes.append((problema, 100, cam["id"], None, jpg.tobytes() if ok else None))
        # Un vértice a menos del 2 % del borde se toma como el borde: una persona cortada por
        # abajo apoya justo en la última fila, y una zona dibujada "hasta abajo" tiene que incluirla.
        pegar = lambda v: 0.0 if v < 0.02 else 1.0 if v > 0.98 else v
        zonas = [np.array([[pegar(x) * ancho, pegar(y) * alto] for x, y in z], np.int32)
                 for z in cam.get("zonas") or []]
        # Fuera del horario de la cámara (ej. sin turno de noche) no se exige casco ni chaleco;
        # zonas prohibidas y caídas siguen siempre
        exige = en_horario(cam.get("horario"), datetime.now())
        if self.exigia and not exige:
            # Terminó el horario: las ventanas de casco y chaleco arrancan de cero mañana. Si la
            # cámara estaba en alerta solo por EPP, la alarma se apaga más abajo (sin alerta a la
            # vista); con alguien en zona prohibida o caído sigue sonando.
            self.seg.ventanas["casco"], self.seg.ventanas["chaleco"] = Ventana(), Ventana()
            self.t_sin_alerta = float("-inf")   # acá no hay parpadeo que esperar: se apaga ya
        self.exigia = exige
        modos = {"casco": exige and bool(cfg["detectar_casco"]), "chaleco": exige and bool(cfg["detectar_chaleco"])}
        caida = bool(cfg["detectar_caida"])
        exentas = [np.array([[x * ancho, y * alto] for x, y in z], np.int32) for z in cam.get("exentas") or []]

        info = {}
        # Hombre caído con modelo de pose y casco o chaleco prendidos: la pose ya corre en analizar()
        # y acostados() no usa `personas`; se ahorra la pasada del modelo de chalecos.
        con_pose = pose.disponible() and (modos["casco"] or modos["chaleco"])
        caras, personas = analizar(img, modos, con_personas=bool(zonas) or (caida and not con_pose),
                                   nivel=self.nivel, info=info,
                                   conocidos=[] if cfg.get("modo_anonimo") else None)   # [] = no identificar a nadie
        # Zonas exentas (oficina, pasillo seguro): ahí no se juzga el EPP de nadie
        exenta = lambda c: any(cv2.pointPolygonTest(z, (c["bbox"][0] + c["bbox"][2] / 2, c["bbox"][1] + c["bbox"][3] / 2),
                                                    False) >= 0 for z in exentas)
        estados = [{"casco": None, "chaleco": None} if exenta(c) else _evaluar(c, cfg["min_confidence"]) for c in caras]
        votos = {k: _peor([e[k] for e in estados]) for k in modos if modos[k]}
        # En zona: el punto de apoyo de la persona (centro del borde de abajo de su caja) cae en un polígono
        en_zona = [p for p in personas
                   if any(cv2.pointPolygonTest(z, (p[0] + p[2] / 2, float(p[1] + p[3])), False) >= 0 for z in zonas)]
        if zonas:
            votos["zona"] = "falta" if en_zona else "ok"
        # Hombre caído: se mide en segundos, así que con modelo de pose se revisa dos veces por
        # segundo y no en cada frame. Si casco o chaleco están prendidos la pose ya corrió en
        # analizar() (info["cuerpos"]); si no, es una pasada más de la red.
        if not caida:
            self.caidos = []
        elif not pose.disponible() or ahora - self.t_pose >= 0.5:
            self.caidos, self.t_pose = acostados(img, personas, caras, modos, info.get("cuerpos"),
                                                 info.get("inclinacion", 0.0)), ahora
        caidos = self.caidos

        nuevas = self.seg.procesar(votos, bool(caidos), int(cfg["confirm_threshold"]),
                                   float(cfg["caida_segundos"]), ahora)
        clase, titulo = self.seg.resumen(list(votos))

        ya_caido = self.seg.caido

        def pintar(im):
            for z in zonas:
                cv2.polylines(im, [z], True, ROJO if en_zona else AMARILLO, 2)
            for p in en_zona:
                _recuadro(im, p, "ZONA PROHIBIDA", ROJO)
            for x, y, w, h in caidos:  # la etiqueta va abajo, para no pisar la de la zona
                _recuadro(im, [x, y + h, w, 0], "HOMBRE CAIDO" if ya_caido else "ACOSTADO", ROJO)
                cv2.rectangle(im, (x, y), (x + w, y + h), ROJO, 2)
            for c, e in zip(caras, estados):
                texto = _etiqueta(c, e)
                if texto:
                    _recuadro(im, c["bbox"], texto, ROJO if "falta" in e.values() else VERDE)

        limpia = None
        if nuevas:   # el cuadro sin dibujar, para revisar la alerta y reentrenar el detector
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            limpia = jpg.tobytes() if ok else None
        self.pintar = pintar
        # Sin detector_casco.onnx no hay cabezas ni alertas de casco: que no parezca una cámara vacía
        sin_detector = modos["casco"] and not helmet_detector.disponible()
        self.estado = {"clase": clase,
                       "titulo": titulo or ("SIN DETECTOR DE CASCOS" if sin_detector else
                                            "ANALIZANDO" if caras or personas or info.get("cuerpos")
                                            else "SIN PERSONAS")}
        if nuevas:
            # La foto solo se usa en una alerta nueva (el video del panel va por vista()):
            # codificar cada cuadro le sacaba CPU al análisis
            pintar(img)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.jpg = jpg.tobytes() if ok else None

        for k in nuevas:
            # Quién fue: en casco y chaleco, las caras en falta; en zona y caída, las caras
            # que están dentro de la caja de esa persona
            cajas = {"zona": en_zona, "caida": caidos}.get(k)
            en_falta = [c for c, e in zip(caras, estados)
                        if (e[k] == "falta" if cajas is None else
                            any(epp._dentro((c["bbox"][0] + c["bbox"][2] / 2, c["bbox"][1] + c["bbox"][3] / 2), p)
                                for p in cajas))]
            persona = ", ".join(sorted({c.get("nombre") or "Sin identificar" for c in en_falta})) or None
            conf = max(en_falta[0]["con_casco"], en_falta[0]["sin_casco"]) if k == "casco" and en_falta else 100
            # Dónde (0 a 1) para el mapa de riesgo: la cabeza en falta, o los pies de la persona en zona / caída
            alto, ancho = img.shape[:2]
            if en_falta:
                x, y, w, h = en_falta[0]["bbox"]
                pos = [(x + w / 2) / ancho, (y + h / 2) / alto]
            elif cajas:
                x, y, w, h = cajas[0]
                pos = [(x + w / 2) / ancho, (y + h) / alto]
            else:
                pos = None
            pos = [min(1.0, max(0.0, v)) for v in pos] if pos else None
            pendiente = _Pendiente((EVENTOS[k], min(100.0, conf), cam["id"], persona, self.jpg, pos))
            pendiente.limpia = limpia
            self.pendientes.append(pendiente)
        # Evento "todos con EPP" cuando la cámara vuelve a verde, como hace el monitor del navegador
        if clase and clase != self.alarma:
            self.alarma = clase
            if clase == "ok":
                self.pendientes.append(("con_casco", 100, cam["id"]))
        if clase is not None or self.alarma != "alerta":
            self.t_sin_alerta = None
        else:
            # Ya no hay ninguna alerta a la vista (zona liberada, caído que se levantó, fin del
            # horario): se apaga la alarma, pero solo si se sostiene. Un parpadeo del detector o un
            # compañero que tapa al caído baja `caido` y reconfirmarlo tarda caida_segundos: con
            # menos espera la sirena se apagaba (y daba luz verde) con el hombre en el suelo.
            # Las de casco y chaleco no llegan acá: su ventana sigue en "falta" hasta ver a alguien con EPP.
            if self.t_sin_alerta is None:
                self.t_sin_alerta = ahora
            if ahora - self.t_sin_alerta >= max(REARME_S, 2 * float(cfg["caida_segundos"])):
                self.alarma, self.t_sin_alerta = None, None
                self.pendientes.append(("con_casco", 100, cam["id"]))
        # Se registran al final y de a uno: si guardar falla (disco, archivo tomado), el evento
        # queda en la lista y se reintenta en el próximo frame en vez de perderse.
        while self.pendientes:
            pendiente = self.pendientes[0]
            evento = registrar_evento(*pendiente)
            self.pendientes.pop(0)
            if getattr(pendiente, "limpia", None) and guardar_muestra and isinstance(evento, dict):
                try:   # si falla, la alerta ya quedó registrada: no se reintenta
                    guardar_muestra(evento, pendiente.limpia)
                except Exception:
                    traceback.print_exc()


# ── Motor ────────────────────────────────────────────────────────────────────

_hilos, _cerrando = {}, []
_tipos = ("relay",)  # el sitio (gunicorn) solo analiza cámaras relay; python app.py además usb e ip
_lock = threading.Lock()


def iniciar():
    """Enciende el motor (lo llama `python src/app.py`). Sin esto sincronizar() no hace nada: un
    servidor sin cámaras propias (el sitio con gunicorn) no abre ningún dispositivo: solo
    analiza cámaras relay (app.py llama a sincronizar() en el primer pedido para vigilarlas)."""
    global _tipos
    _tipos = ("usb", "ip", "relay")
    sincronizar()


def sincronizar():
    """Deja un hilo por cada cámara activa usb/ip/relay. Llamar cuando cambian las cámaras o sus zonas."""
    with _lock:
        camaras = {c["id"]: c for c in data_store.load_cameras()
                   if c.get("active") and c.get("type") in _tipos}
        for cid in list(_hilos):
            if cid not in camaras or not _hilos[cid].is_alive():
                hilo = _hilos.pop(cid)
                hilo.parar.set()
                _cerrando.append(hilo)
        _cerrando[:] = [h for h in _cerrando if h.is_alive()]
        for cid, cam in camaras.items():
            if cid in _hilos:
                _hilos[cid].cam = cam
            else:
                # Espera a los que se están cerrando: dos hilos no pueden abrir la misma cámara USB
                _hilos[cid] = Camara(cam, tuple(_cerrando))
                _hilos[cid].start()


def estado():
    """{id de cámara: {clase, titulo, fps, error}} de las cámaras que analiza el servidor."""
    return {cid: {**h.estado, "fps": round(h.fps, 1), "error": h.error, "lente": h.lente}
            for cid, h in list(_hilos.items())}


def recibir(cam_id, img, numero=None):
    """Cuadro de una cámara relay. False si no hay hilo para ella (borrada o desactivada)."""
    if cam_id not in _hilos:
        sincronizar()
    h = _hilos.get(cam_id)
    if h is None or h.cam.get("type") != "relay":
        return False
    h.recibir(img, numero)
    return True


def cuadro_limpio(cam_id):
    """JPEG del frame más nuevo de la cámara SIN recuadros (para guardarlo y revisarlo), o None."""
    h = _hilos.get(cam_id)
    img = h.crudo if h else None
    if img is None:
        return None
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return jpg.tobytes() if ok else None


def frame(cam_id):
    """JPEG del frame más nuevo de la cámara, con los recuadros del último análisis, o None."""
    h = _hilos.get(cam_id)
    return h.vista() if h else None
