"""Inteligencia de riesgo sobre los eventos guardados (data/events.json).

- puntos():      dónde, dentro de la imagen de cada cámara, pasan las infracciones (mapa de calor)
- matriz():      cuándo: día de la semana × hora, para ver los horarios calientes
- pronostico():  qué horas de mañana tienen más riesgo, según las mismas horas de semanas anteriores
- reporte():     resumen de un día (totales, cámaras, reincidentes, comparación) y texto para mandarlo,
                 redactado por Claude si hay ANTHROPIC_API_KEY o por una plantilla si no.
Todo es Python puro sobre la lista de eventos: no corre modelos ni toca la cámara.
"""
import json
import os
from collections import Counter
from datetime import datetime, timedelta

import data_store

ALERTAS = data_store.ALERT_TYPES
DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
NOMBRES = {"sin_casco": "sin casco", "sin_chaleco": "sin chaleco",
           "zona_prohibida": "en zona prohibida", "hombre_caido": "hombre caído",
           "boton_sos": "botón de pánico", "hombre_muerto": "hombre muerto sin respuesta",
           "posible_desmayo": "posible desmayo", "ritmo_cardiaco": "ritmo cardíaco anormal",
           "wearable_sin_senal": "wearable sin señal"}
MODELO_IA = "claude-opus-5-5"


def _alertas(eventos, desde=None, hasta=None, tipo=None, camara=None):
    """(fecha y hora, evento) de las infracciones en [desde, hasta)."""
    for e in eventos:
        if e.get("type") not in ALERTAS or (tipo and e["type"] != tipo) or (camara and e.get("camera") != camara):
            continue
        try:
            t = datetime.fromisoformat(e["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if (desde is None or t >= desde) and (hasta is None or t < hasta):
            yield t, e


def puntos(eventos, camara, ahora=None, dias=30, tipo=None):
    """[[x, y, tipo], ...] de las alertas con posición de una cámara (por nombre), últimos `dias`."""
    ahora = ahora or datetime.now()
    return [[e["pos"][0], e["pos"][1], e["type"]]
            for _, e in _alertas(eventos, ahora - timedelta(days=dias), None, tipo, camara)
            if isinstance(e.get("pos"), list) and len(e["pos"]) == 2]


def matriz(eventos, ahora=None, dias=28, tipo=None, camara=None):
    """7 filas (lunes a domingo) × 24 horas con la cantidad de alertas de los últimos `dias`."""
    ahora = ahora or datetime.now()
    m = [[0] * 24 for _ in range(7)]
    for t, _ in _alertas(eventos, ahora - timedelta(days=dias), None, tipo, camara):
        m[t.weekday()][t.hour] += 1
    return m


def pronostico(eventos, ahora=None, semanas=4):
    """Alertas esperadas por hora para mañana: promedio de esa hora en el mismo día de las últimas
    `semanas` semanas, con más peso a las recientes (4, 3, 2, 1).
    shortcut: promedio ponderado, sin estacionalidad ni clima; con meses de datos, probar un modelo de conteo."""
    ahora = ahora or datetime.now()
    manana = (ahora + timedelta(days=1)).date()
    horas = [0.0] * 24
    pesos = 0
    for s in range(1, semanas + 1):
        dia = manana - timedelta(weeks=s)
        inicio = datetime.combine(dia, datetime.min.time())
        peso = semanas + 1 - s
        pesos += peso
        for t, _ in _alertas(eventos, inicio, inicio + timedelta(days=1)):
            horas[t.hour] += peso
    horas = [round(h / pesos, 2) if pesos else 0.0 for h in horas]
    picos = sorted((h for h in range(24) if horas[h] > 0), key=lambda h: (-horas[h], h))[:3]
    return {"fecha": manana.isoformat(), "dia": DIAS[manana.weekday()], "horas": horas,
            "picos": [{"hora": h, "esperadas": horas[h]} for h in picos], "semanas_con_datos": pesos > 0}


def tendencia(eventos, ahora=None):
    """{tipo: [últimos 7 días, 7 días anteriores]} de cada tipo de alerta."""
    ahora = ahora or datetime.now()
    t = {k: [0, 0] for k in ALERTAS}
    for ts, e in _alertas(eventos, ahora - timedelta(days=14), ahora):
        t[e["type"]][0 if ts >= ahora - timedelta(days=7) else 1] += 1
    return t


TURNOS = (("Mañana", 6, 14), ("Tarde", 14, 22), ("Noche", 22, 6))
CORRECCION_MAX = timedelta(minutes=30)   # una vuelta a verde más tarde ya no cuenta como corrección


def turno(t):
    for nombre, desde, hasta in TURNOS:
        if (desde <= t.hour < hasta) if desde < hasta else (t.hour >= desde or t.hour < hasta):
            return nombre


def kpis(eventos, ahora=None, dias=30):
    """Indicadores HSE para auditoría (ISO 45001) de los últimos `dias`:
    - por turno: alertas, por tipo y minutos hasta la corrección (mediana del tiempo entre la alerta
      y el siguiente "todos con EPP" de la misma cámara, si llega en menos de 30 min)
    - racha de días sin alertas y alertas por día."""
    ahora = ahora or datetime.now()
    desde = ahora - timedelta(days=dias)
    verdes = {}
    for e in eventos:
        if e.get("type") == "con_casco":
            try:
                verdes.setdefault(e.get("camera"), []).append(datetime.fromisoformat(e["timestamp"]))
            except (KeyError, TypeError, ValueError):
                pass
    for v in verdes.values():
        v.sort()
    filas = {n: {"turno": n, "alertas": 0, "por_tipo": Counter(), "correcciones": []} for n, _, _ in TURNOS}
    por_dia = Counter()
    for t, e in _alertas(eventos, desde, ahora):
        f = filas[turno(t)]
        f["alertas"] += 1
        f["por_tipo"][e["type"]] += 1
        por_dia[t.date().isoformat()] += 1
        siguiente = next((v for v in verdes.get(e.get("camera"), []) if v > t), None)
        if siguiente and siguiente - t <= CORRECCION_MAX:
            f["correcciones"].append((siguiente - t).total_seconds() / 60)
    for f in filas.values():
        c = sorted(f.pop("correcciones"))
        f["correccion_min"] = round(c[len(c) // 2], 1) if c else None
        f["corregidas"] = len(c)
        f["por_tipo"] = dict(f["por_tipo"])
    racha, d = 0, ahora.date()
    while racha < dias and not por_dia.get((d - timedelta(days=racha)).isoformat()):
        racha += 1
    return {"dias": dias, "total": sum(por_dia.values()), "turnos": list(filas.values()),
            "dias_sin_alertas": racha,
            "por_dia": [{"fecha": (desde.date() + timedelta(days=i + 1)).isoformat(),
                         "alertas": por_dia.get((desde.date() + timedelta(days=i + 1)).isoformat(), 0)}
                        for i in range(dias)]}


def reporte(eventos, dia=None):
    """Resumen de un día (date): lo que se manda a la mañana siguiente."""
    dia = dia or (datetime.now() - timedelta(days=1)).date()
    inicio = datetime.combine(dia, datetime.min.time())
    del_dia = list(_alertas(eventos, inicio, inicio + timedelta(days=1)))
    semana_antes = sum(1 for _ in _alertas(eventos, inicio - timedelta(days=7), inicio - timedelta(days=6)))
    personas = Counter()
    for _, e in del_dia:
        for p in str(e.get("persona") or "").split(", "):
            if p and p != "Sin identificar":
                personas[p] += 1
    horas = Counter(t.hour for t, _ in del_dia)
    con_epp = sum(1 for e in eventos if e.get("type") == "con_casco"
                  and str(e.get("timestamp", "")).startswith(dia.isoformat()))
    return {
        "fecha": dia.isoformat(),
        "dia": DIAS[dia.weekday()],
        "total": len(del_dia),
        "mismo_dia_semana_anterior": semana_antes,
        "por_tipo": dict(Counter(e["type"] for _, e in del_dia)),
        "por_camara": dict(Counter(e.get("camera", "") for _, e in del_dia).most_common(5)),
        "reincidentes": [{"persona": p, "alertas": n} for p, n in personas.most_common(5) if n >= 2],
        "hora_pico": ({"hora": horas.most_common(1)[0][0], "alertas": horas.most_common(1)[0][1]}
                      if horas else None),
        "vueltas_a_verde": con_epp,
        "pronostico_manana": pronostico(eventos, inicio + timedelta(hours=23, minutes=59)),
    }


def texto(r):
    """Reporte en texto llano, sin IA (también es el respaldo si la IA falla)."""
    lineas = [f"RAC Vision · reporte del {r['dia']} {r['fecha']}"]
    if not r["total"]:
        lineas.append("Sin alertas de seguridad en el día. 👍")
    else:
        dif = r["total"] - r["mismo_dia_semana_anterior"]
        comp = ("igual que" if dif == 0 else f"{abs(dif)} {'más' if dif > 0 else 'menos'} que") + \
            f" el {r['dia']} anterior ({r['mismo_dia_semana_anterior']})"
        lineas.append(f"{r['total']} alertas, {comp}.")
        lineas.append("Por tipo: " + ", ".join(f"{n} {NOMBRES.get(k, k)}" for k, n in r["por_tipo"].items()) + ".")
        lineas.append("Cámaras: " + ", ".join(f"{c} ({n})" for c, n in r["por_camara"].items()) + ".")
        if r["hora_pico"]:
            lineas.append(f"Hora pico: {r['hora_pico']['hora']:02d}:00 ({r['hora_pico']['alertas']} alertas).")
        if r["reincidentes"]:
            lineas.append("Reincidentes: " + ", ".join(f"{x['persona']} ({x['alertas']})" for x in r["reincidentes"])
                          + ". Conviene una charla de seguridad.")
    p = r["pronostico_manana"]
    if p["picos"]:
        horas = [f"{h:02d}:00" for h in sorted(x["hora"] for x in p["picos"])]
        lista = horas[0] if len(horas) == 1 else ", ".join(horas[:-1]) + " y " + horas[-1]
        lineas.append(f"Mañana ({p['dia']}) ojo a las {lista}: ahí se concentran las alertas.")
    return "\n".join(lineas)


def redactar(r):
    """Texto del reporte escrito por Claude a partir de los números; sin ANTHROPIC_API_KEY, la plantilla.
    Solo se mandan conteos y nombres de cámaras / personas ya guardados: nunca fotos."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return texto(r)
    try:
        import anthropic
        client = anthropic.Anthropic(timeout=60.0)
        respuesta = client.beta.messages.create(
            model=MODELO_IA,
            max_tokens=2000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",   # si el modelo declina, lo sigue otro en la misma llamada
            system=("Sos el asistente de seguridad e higiene de una obra que usa RAC Vision (cámaras que "
                    "detectan casco, chaleco, zonas prohibidas y caídas). Escribís el reporte diario para el "
                    "jefe de obra, en español rioplatense, por Telegram: máximo 120 palabras, sin markdown, "
                    "con 1 a 3 acciones concretas para mañana. Usá solo los datos que te pasan; no inventes."),
            messages=[{"role": "user", "content": "Datos del día:\n" + json.dumps(r, ensure_ascii=False)}],
        )
        if respuesta.stop_reason == "refusal":
            return texto(r)
        escrito = "".join(b.text for b in respuesta.content if b.type == "text").strip()
        return escrito or texto(r)
    except Exception as e:  # sin red, clave inválida, cuota: el reporte sale igual con la plantilla
        print(f"[!] Reporte con IA falló, se usa la plantilla: {e}", flush=True)
        return texto(r)


if __name__ == "__main__":  # python riesgo.py → reporte de ayer en la consola
    print(redactar(reporte(data_store.load_events())))
