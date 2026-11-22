"""Wearables: pulsera o arnés con pulso, acelerómetro y botón que manda telemetría a RAC.

- Registro de dispositivos (data/dispositivos.json): cada uno tiene un trabajador asignado y un
  token propio. El token se muestra una sola vez al crearlo; acá solo se guarda su hash.
- Reglas de alerta (`Reglas.evaluar`), funciones puras sobre cada lectura:
    boton_sos          el trabajador mantuvo apretado el botón (pide ayuda)
    hombre_muerto      el equipo avisó que estuvo quieto, vibró y nadie apretó el botón a tiempo
    posible_desmayo    quieto después de una caída, o quieto con el pulso bajo o sin pulso
    ritmo_cardiaco     pulso fuera de rango, sin pulso con el sensor apoyado, u oxígeno bajo, sostenido
    wearable_sin_senal el equipo dejó de mandar datos (batería, golpe, fuera de cobertura)
- Aprendizaje: cuando un supervisor marca una alerta de pulso o desmayo como falsa alarma, los
  umbrales de ese trabajador se ajustan (`aprender_de_falsa`), con topes para no apagar la alerta.

El firmware de referencia está en firmware/wearable_rac/ y simulador_wearable.py prueba todo sin hardware.
"""
import hashlib
import secrets
import threading
import uuid
from datetime import datetime

import data_store

ARCHIVO = "dispositivos.json"

# Tipos de alerta que generan los wearables (también están en data_store.ALERT_TYPES)
TIPOS = ("boton_sos", "hombre_muerto", "posible_desmayo", "ritmo_cardiaco", "wearable_sin_senal")
# Los que vale la pena cruzar con la cámara (persona en el piso)
TIPOS_DE_CAIDA = ("posible_desmayo", "hombre_muerto", "boton_sos")

UMBRALES = {
    "bpm_min": 40,        # bradicardia: menos de esto, sostenido
    "bpm_max": 160,       # taquicardia: más de esto, sostenido (un minero trabajando duro puede pasar 140)
    "spo2_min": 88,       # oxígeno en sangre (%)
    "sostenido_s": 20,    # cuánto tiene que durar un pulso anormal para avisar
    "inactivo_s": 30,     # quieto este tiempo después de una caída = posible desmayo
}
# Lo que el aprendizaje nunca va a superar: aunque marquen falsas, la alerta sigue existiendo
TOPES = {"bpm_min": 30, "bpm_max": 200, "inactivo_s": 120}
QUIETO = 0.05             # actividad (g RMS) por debajo de esto = el cuerpo no se mueve
CAIDA_VALIDA_S = 90       # una caída cuenta para el desmayo durante este tiempo
SIN_SENAL_S = 90          # sin datos este tiempo = alerta
CONFIANZA = {"boton_sos": 100, "hombre_muerto": 100, "posible_desmayo": 85,
             "ritmo_cardiaco": 80, "wearable_sin_senal": 100}

_lock = threading.RLock()


# ═══════════════════════════════════════
#  REGISTRO DE DISPOSITIVOS
# ═══════════════════════════════════════

def _hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def cargar():
    return data_store._load_json(ARCHIVO, [])


def guardar(dispositivos):
    data_store._save_json(ARCHIVO, dispositivos)


def crear(nombre, persona, camara=""):
    """Registra un equipo. Devuelve (dispositivo, token): el token no se vuelve a poder ver."""
    token = "rac_" + secrets.token_urlsafe(24)
    disp = {
        "id": str(uuid.uuid4()),
        "nombre": nombre.strip()[:60],
        "persona": persona.strip()[:120],
        "camara": (camara or "").strip()[:120],   # cámara de la zona donde trabaja (para cruzar caídas)
        "token_hash": _hash(token),
        "activo": True,
        "umbrales": {},                           # lo que aprendió para este trabajador
        "aprendido": [],
        "creado": datetime.now().isoformat(timespec="seconds"),
    }
    with _lock:
        lista = cargar()
        lista.append(disp)
        guardar(lista)
    return disp, token


def regenerar_token(disp_id):
    token = "rac_" + secrets.token_urlsafe(24)
    with _lock:
        lista = cargar()
        for d in lista:
            if d["id"] == disp_id:
                d["token_hash"] = _hash(token)
                guardar(lista)
                return token
    return None


def borrar(disp_id):
    with _lock:
        lista = cargar()
        quedan = [d for d in lista if d["id"] != disp_id]
        guardar(quedan)
    _estados.pop(disp_id, None)
    return len(quedan) < len(lista)


def por_token(token):
    """El dispositivo activo con ese token, o None. Compara hashes en tiempo constante."""
    if not isinstance(token, str) or not token.startswith("rac_"):
        return None
    h = _hash(token)
    for d in cargar():
        if d.get("activo", True) and secrets.compare_digest(d.get("token_hash", ""), h):
            return d
    return None


def obtener(disp_id):
    for d in cargar():
        if d["id"] == disp_id:
            return d
    return None


def umbrales_de(disp):
    return {**UMBRALES, **(disp.get("umbrales") or {})}


def restablecer_umbrales(disp_id):
    with _lock:
        lista = cargar()
        for d in lista:
            if d["id"] == disp_id:
                d["umbrales"], d["aprendido"] = {}, []
                guardar(lista)
                return True
    return False


# ═══════════════════════════════════════
#  LECTURAS
# ═══════════════════════════════════════

def _numero(v, minimo, maximo):
    return (isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and minimo <= v <= maximo)


def validar_lectura(data):
    """(lectura normalizada, None) o (None, error). Todos los campos son opcionales salvo que
    venga al menos uno; lo que no se manda se toma como "no se sabe"."""
    if not isinstance(data, dict):
        return None, "Se esperaba un objeto JSON"
    lectura = {}
    for campo, lo, hi in (("bpm", 0, 250), ("spo2", 50, 100), ("bateria", 0, 100),
                          ("actividad", 0, 16), ("temp", 20, 45)):
        v = data.get(campo)
        if v is None:
            continue
        if not _numero(v, lo, hi):
            return None, f"'{campo}' debe ser un número entre {lo} y {hi}"
        lectura[campo] = round(float(v), 2)
    for campo in ("contacto", "caida", "sos", "hombre_muerto"):
        v = data.get(campo)
        if v is None:
            continue
        if not isinstance(v, bool):
            return None, f"'{campo}' debe ser true o false"
        lectura[campo] = v
    if not lectura:
        return None, "La lectura no trae ningún dato"
    return lectura, None


# ═══════════════════════════════════════
#  REGLAS DE ALERTA
# ═══════════════════════════════════════

class Estado:
    """Lo que hace falta recordar de un equipo entre lecturas (en memoria del proceso)."""

    def __init__(self):
        self.desde = {}          # condición -> momento en que empezó (segundos)
        self.activas = set()     # alertas ya avisadas que siguen pasando (no se repiten)
        self.ultima_caida = None
        self.ultima = None       # (momento, lectura) de la última lectura
        self.sin_senal = False


def _sostenida(estado, clave, ocurre, ahora, segundos):
    """True cuando la condición lleva `segundos` pasando sin cortarse."""
    if not ocurre:
        estado.desde.pop(clave, None)
        return False
    inicio = estado.desde.setdefault(clave, ahora)
    return ahora - inicio >= segundos


def evaluar(estado, lectura, ahora, umbrales=None):
    """Aplica las reglas a una lectura. Devuelve [(tipo, detalle), ...] con las alertas NUEVAS
    (una alerta que sigue pasando no se repite hasta que se corte)."""
    u = {**UMBRALES, **(umbrales or {})}
    estado.ultima = (ahora, lectura)
    estado.sin_senal = False
    bpm, spo2 = lectura.get("bpm"), lectura.get("spo2")
    contacto = lectura.get("contacto", True)
    actividad = lectura.get("actividad")
    quieto = actividad is not None and actividad < QUIETO
    if lectura.get("caida"):
        estado.ultima_caida = ahora
    cayo_hace_poco = estado.ultima_caida is not None and ahora - estado.ultima_caida <= CAIDA_VALIDA_S

    sin_pulso = contacto is not False and bpm == 0   # 0 = el sensor está apoyado y no encuentra latidos
    bradi = bpm is not None and 0 < bpm < u["bpm_min"]
    taqui = bpm is not None and bpm > u["bpm_max"]
    hipoxia = spo2 is not None and spo2 < u["spo2_min"]

    condiciones = {
        "boton_sos": (bool(lectura.get("sos")), "botón de pánico"),
        "hombre_muerto": (bool(lectura.get("hombre_muerto")), "no respondió a la vibración"),
        "posible_desmayo": (
            _sostenida(estado, "quieto", quieto and (cayo_hace_poco or sin_pulso or bradi), ahora, u["inactivo_s"]),
            "quieto después de una caída" if cayo_hace_poco else "quieto con el pulso bajo"),
    }
    detalle_ritmo = ("sin pulso" if sin_pulso else f"bradicardia {bpm:.0f} lpm" if bradi
                     else f"taquicardia {bpm:.0f} lpm" if taqui else f"oxígeno {spo2:.0f}%" if hipoxia else "")
    condiciones["ritmo_cardiaco"] = (
        _sostenida(estado, "ritmo", bool(detalle_ritmo), ahora, u["sostenido_s"]), detalle_ritmo)

    nuevas = []
    for tipo, (ocurre, detalle) in condiciones.items():
        if ocurre and tipo not in estado.activas:
            estado.activas.add(tipo)
            nuevas.append((tipo, detalle))
        elif not ocurre:
            estado.activas.discard(tipo)
    estado.activas.discard("wearable_sin_senal")
    return nuevas


def callados(estados, ahora, limite=SIN_SENAL_S):
    """Ids de los equipos que dejaron de mandar datos hace más de `limite` segundos
    (cada uno una sola vez, hasta que vuelva a mandar)."""
    ids = []
    for disp_id, est in estados.items():
        if est.ultima and not est.sin_senal and ahora - est.ultima[0] > limite:
            est.sin_senal = True
            ids.append(disp_id)
    return ids


_estados = {}   # id de dispositivo -> Estado. shortcut: por proceso (gunicorn corre con un worker)


def estado_de(disp_id):
    with _lock:
        return _estados.setdefault(disp_id, Estado())


def resumen(disp, ahora):
    """Lo que muestra el panel de un equipo: última lectura, hace cuánto y en qué estado está."""
    est = _estados.get(disp["id"])
    ultima = est.ultima if est else None
    if not ultima:
        situacion = "sin_datos"
    elif est.sin_senal or ahora - ultima[0] > SIN_SENAL_S:
        situacion = "sin_senal"
    elif est.activas:
        situacion = "alerta"
    else:
        situacion = "ok"
    return {
        "id": disp["id"], "nombre": disp["nombre"], "persona": disp["persona"], "camara": disp.get("camara", ""),
        "activo": disp.get("activo", True), "situacion": situacion,
        "alertas": sorted(est.activas) if est else [],
        "lectura": ultima[1] if ultima else None,
        "hace_s": round(ahora - ultima[0]) if ultima else None,
        "umbrales": umbrales_de(disp), "aprendido": disp.get("aprendido", [])[-5:],
    }


# ═══════════════════════════════════════
#  CRUCE CON LA CÁMARA
# ═══════════════════════════════════════

def buscar_confirmacion(eventos, tipos, momento, ventana_s=120, camara=None, persona=None):
    """El evento más reciente de `tipos` a menos de `ventana_s` de `momento` (datetime), o None.
    Si se da cámara, tiene que ser de esa cámara; si se da persona, tiene que ser esa persona o
    alguien sin identificar (la cámara no siempre reconoce la cara de alguien tirado en el piso)."""
    for e in eventos:   # vienen del más nuevo al más viejo
        if e.get("type") not in tipos:
            continue
        try:
            t = datetime.fromisoformat(e["timestamp"])
        except (KeyError, ValueError):
            continue
        if abs((momento - t).total_seconds()) > ventana_s:
            if t < momento:
                break
            continue
        if camara and e.get("camera") != camara:
            continue
        if persona and e.get("persona") not in (persona, "Sin identificar", None):
            continue
        return e
    return None


# ═══════════════════════════════════════
#  APRENDIZAJE CON LAS ALERTAS MARCADAS
# ═══════════════════════════════════════

def aprender_de_falsa(evento):
    """Un supervisor dijo que esta alerta de wearable fue falsa: se aflojan los umbrales de ese
    trabajador justo lo necesario para que la misma lectura no vuelva a avisar. Devuelve el
    cambio aplicado (texto) o None."""
    disp_id, medicion, tipo = evento.get("dispositivo"), evento.get("medicion") or {}, evento.get("type")
    if not disp_id or tipo not in ("ritmo_cardiaco", "posible_desmayo"):
        return None
    with _lock:
        lista = cargar()
        disp = next((d for d in lista if d["id"] == disp_id), None)
        if not disp:
            return None
        u = umbrales_de(disp)
        disp.setdefault("umbrales", {})
        bpm, detalle = medicion.get("bpm"), evento.get("detalle", "")
        cambio = None
        if tipo == "ritmo_cardiaco" and detalle.startswith("taquicardia") and bpm and u["bpm_max"] < TOPES["bpm_max"]:
            nuevo = min(TOPES["bpm_max"], max(u["bpm_max"], int(bpm) + 5))
            if nuevo != u["bpm_max"]:
                disp["umbrales"]["bpm_max"] = nuevo
                cambio = f"Taquicardia: avisa desde {nuevo} lpm (antes {u['bpm_max']})"
        elif tipo == "ritmo_cardiaco" and detalle.startswith("bradicardia") and bpm and u["bpm_min"] > TOPES["bpm_min"]:
            nuevo = max(TOPES["bpm_min"], min(u["bpm_min"], int(bpm) - 3))
            if nuevo != u["bpm_min"]:
                disp["umbrales"]["bpm_min"] = nuevo
                cambio = f"Bradicardia: avisa por debajo de {nuevo} lpm (antes {u['bpm_min']})"
        elif tipo == "posible_desmayo" and u["inactivo_s"] < TOPES["inactivo_s"]:
            nuevo = min(TOPES["inactivo_s"], u["inactivo_s"] + 15)
            disp["umbrales"]["inactivo_s"] = nuevo
            cambio = f"Desmayo: espera {nuevo} s quieto (antes {u['inactivo_s']} s)"
        if not cambio:
            return None
        disp.setdefault("aprendido", []).append(
            {"fecha": datetime.now().isoformat(timespec="seconds"), "cambio": cambio, "evento": evento.get("id")})
        disp["aprendido"] = disp["aprendido"][-50:]
        guardar(lista)
        return cambio
