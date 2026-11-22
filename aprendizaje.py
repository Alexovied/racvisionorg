"""Aprendizaje con las alertas que los supervisores marcan como reales o falsas.

Cada alerta del panel se puede marcar como "Fue real" o "Falsa alarma" (data_store.set_feedback).
Con esas marcas:
- precision(): qué porcentaje de las alertas de cada tipo fueron reales, para saber en qué
  confiar y qué revisar.
- sugerir_confianza(): para las alertas de cámara, la confianza mínima que habría descartado
  la mayor cantidad de falsas alarmas sin perder ninguna alerta real. Es una sugerencia: el
  admin decide si la aplica en Configuración.
- Los wearables ajustan sus propios umbrales por trabajador (wearables.aprender_de_falsa).
- Las fotos marcadas sirven de ejemplos para reentrenar el detector (exportar_fotos).
"""
import os
import shutil

import data_store

# Alertas que salen de la cámara y dependen de la confianza del modelo
TIPOS_DE_CAMARA = ("sin_casco", "sin_chaleco", "zona_prohibida", "hombre_caido")
MINIMO_DE_MARCAS = 10      # menos marcas que esto no alcanzan para sugerir nada


def marcadas(eventos, tipo=None):
    return [e for e in eventos if (e.get("feedback") or {}).get("veredicto") in ("real", "falsa")
            and (tipo is None or e.get("type") == tipo)]


def precision(eventos):
    """{tipo: {"reales": n, "falsas": n, "precision": % o None}} de los tipos con alguna marca."""
    tabla = {}
    for e in marcadas(eventos):
        fila = tabla.setdefault(e["type"], {"reales": 0, "falsas": 0})
        fila["reales" if e["feedback"]["veredicto"] == "real" else "falsas"] += 1
    for fila in tabla.values():
        total = fila["reales"] + fila["falsas"]
        fila["precision"] = round(100 * fila["reales"] / total) if total else None
    return tabla


def sugerir_confianza(eventos, actual, tipos=TIPOS_DE_CAMARA):
    """Confianza mínima sugerida (50 a 99) o None si no hay datos suficientes o no conviene
    cambiar. Es el valor más alto que deja pasar TODAS las alertas reales marcadas."""
    datos = [(float(e.get("confidence") or 0), e["feedback"]["veredicto"] == "real")
             for e in marcadas(eventos) if e.get("type") in tipos]
    if len(datos) < MINIMO_DE_MARCAS:
        return None
    reales = [c for c, real in datos if real]
    falsas = [c for c, real in datos if not real]
    if not reales or not falsas:
        return None
    umbral = max(50, min(99, int(min(reales))))   # sin perder ninguna real
    if umbral <= actual:
        return None
    filtradas = sum(1 for c in falsas if c < umbral)
    if not filtradas:
        return None
    return {"confianza": umbral, "falsas_evitadas": filtradas, "falsas": len(falsas), "reales": len(reales)}


def exportar_fotos(eventos, destino):
    """Copia las fotos de las alertas marcadas a destino/<tipo>/{real,falsa}/<id>.<ext>, para
    sumarlas al dataset de reentrenamiento (las falsas son los ejemplos que el modelo confundió).
    Devuelve cuántas copió."""
    n = 0
    for e in marcadas(eventos):
        foto = data_store.existing_alert_photo(e["id"]) if e.get("foto") else None
        if not foto:
            continue
        carpeta = os.path.join(destino, e["type"], e["feedback"]["veredicto"])
        os.makedirs(carpeta, exist_ok=True)
        shutil.copy2(foto[0], os.path.join(carpeta, os.path.basename(foto[0])))
        n += 1
    return n


if __name__ == "__main__":   # python aprendizaje.py dataset_feedback/
    import sys
    destino = sys.argv[1] if len(sys.argv) > 1 else "dataset_feedback"
    print(f"{exportar_fotos(data_store.load_events(), destino)} fotos copiadas a {destino}/")
