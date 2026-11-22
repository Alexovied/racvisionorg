"""Benchmark de consumo de RAC bajo carga: CPU, RAM, frames por segundo y latencia.

Simula N cámaras como lo hace el navegador en /app: cada una manda un frame
640x480 JPEG a /predict, espera la respuesta y espera 100 ms más (scan_rate_ms).
Mide el contenedor con `docker stats`, así que corre en el servidor (solo usa la
biblioteca estándar de Python). Se usa contra un contenedor de prueba, nunca contra
producción:

    python3 benchmark_carga.py URL CONTENEDOR CARPETA_FRAMES SALIDA.json [cámaras...]

La contraseña del usuario admin de prueba se lee de la variable RAC_BENCH_PASSWORD.
"""
import base64
import glob
import http.cookiejar
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

PAUSA = 0.1        # scan_rate_ms del panel
DURACION = 40      # segundos medidos por nivel de carga
CALENTAMIENTO = 8  # segundos que no se miden al empezar cada nivel


def abrir_sesion(url):
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    datos = urllib.parse.urlencode({"username": "admin", "password": os.environ["RAC_BENCH_PASSWORD"]}).encode()
    op.open(url + "/admin/login", datos, timeout=30)
    assert any(c.name == "session" for c in jar), "no se pudo iniciar sesión"
    return op


def leer_stats(contenedor):
    """(cpu %, ram MiB) del contenedor. 100 % = un núcleo entero."""
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}};{{.MemUsage}}", contenedor],
                         capture_output=True, text=True).stdout.strip()
    cpu, mem = out.split(";")
    valor, unidad = mem.split("/")[0].strip()[:-3], mem.split("/")[0].strip()[-3:]
    return float(cpu.rstrip("%")), float(valor) * (1024 if unidad == "GiB" else 1)


def camara(op, url, cuerpos, fin, resultados, i):
    n = i
    while time.time() < fin:
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(url + "/predict", cuerpos[n % len(cuerpos)], {"Content-Type": "application/json"})
            ok = op.open(req, timeout=60).status == 200
        except Exception:
            ok = False
        resultados.append((time.time(), time.perf_counter() - t0, ok))
        n += 1
        time.sleep(PAUSA)


def nivel(op, url, contenedor, cuerpos, camaras):
    inicio = time.time()
    medir_desde, fin = inicio + CALENTAMIENTO, inicio + CALENTAMIENTO + DURACION
    resultados, stats = [], []
    hilos = [threading.Thread(target=camara, args=(op, url, cuerpos, fin, resultados, i)) for i in range(camaras)]
    for h in hilos:
        h.start()
    while time.time() < fin:
        s = leer_stats(contenedor)
        if time.time() >= medir_desde:
            stats.append(s)
    for h in hilos:
        h.join()
    medidos = [(lat, ok) for t, lat, ok in resultados if t >= medir_desde]
    lat = sorted(l for l, ok in medidos if ok)
    cpu, ram = [s[0] for s in stats], [s[1] for s in stats]
    return {
        "camaras": camaras,
        "fps_total": round(len(lat) / DURACION, 2),
        "fps_por_camara": round(len(lat) / DURACION / camaras, 2),
        "latencia_p50_ms": round(1000 * statistics.median(lat)),
        "latencia_p95_ms": round(1000 * lat[int(0.95 * (len(lat) - 1))]),
        "errores": sum(1 for _, ok in medidos if not ok),
        "cpu_media_pct": round(statistics.mean(cpu)), "cpu_max_pct": round(max(cpu)),
        "ram_media_mib": round(statistics.mean(ram)), "ram_max_mib": round(max(ram)),
    }


def main():
    url, contenedor, carpeta, salida = sys.argv[1:5]
    niveles = [int(x) for x in sys.argv[5:]] or [1, 2, 4, 8]
    cuerpos = [json.dumps({"image": "data:image/jpeg;base64," + base64.b64encode(open(f, "rb").read()).decode()}).encode()
               for f in sorted(glob.glob(os.path.join(carpeta, "*.jpg")))]
    assert cuerpos, "no hay frames"
    op = abrir_sesion(url)
    reposo = [leer_stats(contenedor) for _ in range(5)]
    res = {"reposo": {"cpu_media_pct": round(statistics.mean(s[0] for s in reposo), 1),
                      "ram_media_mib": round(statistics.mean(s[1] for s in reposo))}, "niveles": []}
    print("reposo:", res["reposo"], flush=True)
    for n in niveles:
        r = nivel(op, url, contenedor, cuerpos, n)
        res["niveles"].append(r)
        print(r, flush=True)
    json.dump(res, open(salida, "w"), indent=1)


if __name__ == "__main__":
    main()
