"""Simulador de wearable: manda telemetría a RAC como lo haría la pulsera, para probar sin hardware.

1. En el panel, Wearables → Registrar un equipo. Copiar el token que aparece.
2. python src/simulador_wearable.py --token rac_XXXX --escenario desmayo
   (contra el servidor local; para otro: --url https://...)
3. Mirar el panel: la tarjeta del equipo cambia de estado y aparece el aviso rojo arriba.

Escenarios:
  normal        pulso de trabajo (80-110 lpm), moviéndose. No debería avisar nunca.
  sos           a los 10 s el trabajador mantiene apretado el botón de pánico.
  hombre_muerto a los 10 s queda quieto; el equipo vibra y nadie aprieta el botón.
  desmayo       a los 10 s se cae y queda quieto, con el pulso bajando.
  taquicardia   el pulso sube a 175 lpm y se queda ahí.
  sin_pulso     el sensor sigue apoyado pero no encuentra latidos.
  sin_senal     manda unos datos y se calla (el servidor avisa a los 90 s).

Las reglas que deciden cuándo avisar están en el servidor (wearables.py): el simulador solo
manda lo que mediría el equipo.
"""
import argparse
import random
import sys
import time

import requests

ESCENARIOS = ("normal", "sos", "hombre_muerto", "desmayo", "taquicardia", "sin_pulso", "sin_senal")


def lectura(escenario, t, bateria):
    """Lo que mandaría la pulsera en el segundo t del escenario."""
    l = {"bpm": random.randint(80, 110), "spo2": random.randint(95, 99), "contacto": True,
         "actividad": round(random.uniform(0.2, 0.8), 2), "bateria": bateria}
    if t < 10 or escenario == "normal":
        return l
    if escenario == "sos":
        l["sos"] = t < 16
    elif escenario == "hombre_muerto":
        l["actividad"] = 0.01
        l["hombre_muerto"] = t >= 25   # en el equipo: 10 s quieto + 5 s de vibración sin respuesta (acá más lento)
    elif escenario == "desmayo":
        l["caida"] = t < 12
        l["actividad"] = 0.01
        l["bpm"] = max(38, 90 - 2 * (t - 10))
        l["spo2"] = max(86, 97 - (t - 10) // 4)
    elif escenario == "taquicardia":
        l["bpm"] = random.randint(172, 180)
    elif escenario == "sin_pulso":
        l["bpm"] = 0
        l["spo2"] = None
        l["actividad"] = 0.02
    return {k: v for k, v in l.items() if v is not None}


def main(argv=None):
    p = argparse.ArgumentParser(description="Simula una pulsera RAC mandando telemetría.")
    p.add_argument("--url", default="http://localhost:5000", help="dirección del panel (sin / al final)")
    p.add_argument("--token", required=True, help="token del equipo (Wearables → Registrar un equipo)")
    p.add_argument("--escenario", choices=ESCENARIOS, default="normal")
    p.add_argument("--intervalo", type=float, default=2.0, help="segundos entre lecturas (la pulsera usa 5)")
    p.add_argument("--duracion", type=float, default=60.0, help="segundos que dura la simulación")
    a = p.parse_args(argv)

    destino = a.url.rstrip("/") + "/admin/api/dispositivos/telemetria"
    headers = {"Authorization": f"Bearer {a.token}"}
    inicio, bateria = time.monotonic(), 87
    print(f"Simulando «{a.escenario}» contra {destino}. Ctrl+C para cortar.")
    while (t := time.monotonic() - inicio) < a.duracion:
        if a.escenario == "sin_senal" and t > 10:
            print("… el equipo se calló. El panel avisa cuando pasen 90 s sin datos.")
            break
        datos = lectura(a.escenario, t, bateria)
        try:
            r = requests.post(destino, json=datos, headers=headers, timeout=5)
        except requests.RequestException as e:
            print(f"[!] Sin conexión con el panel: {e}")
            return 1
        if r.status_code == 401:
            print("[!] Token inválido: registrá el equipo en el panel y copiá el token que aparece.")
            return 1
        respuesta = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
        alertas = respuesta.get("alertas") if isinstance(respuesta, dict) else None
        print(f"{t:5.1f}s  {datos}  →  {r.status_code}" + (f"  ⚠ ALERTA: {', '.join(alertas)}" if alertas else ""))
        time.sleep(a.intervalo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
