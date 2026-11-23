"""Reenvía una cámara de la red local al servidor RAC, que la analiza (cámara tipo "relay").

La PC no analiza nada: lee la cámara, achica el cuadro y manda unos pocos por segundo.
La cámara no queda expuesta a internet: es la PC la que sale hacia el servidor.

    python herramientas/relay_camara.py --camara rtsp://10.10.10.124:554/onvif1 \
        --servidor https://dev.racvision.org --id <ID de la cámara> --token rac_...

El ID y el token los da el panel al registrar la cámara como "Cámara IP en otra red (relay)".
El token también se puede pasar en la variable RAC_TOKEN, para que no quede en el historial.
"""
import argparse
import os
import threading
import time

import cv2
import requests


class Lector(threading.Thread):
    """Lee la cámara sin parar y se queda solo con el último cuadro: si se leyera al ritmo del
    envío, el búfer del RTSP se llena y se mandan cuadros de hace varios segundos."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.cuadro = url, None

    def run(self):
        while True:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000])
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                self.cuadro = img
            cap.release()
            self.cuadro = None
            print("[!] Sin señal de la cámara, reintento en 5 s", flush=True)
            time.sleep(5)


def jpeg(img, ancho):
    if img.shape[1] > ancho:
        img = cv2.resize(img, (ancho, round(img.shape[0] * ancho / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else None


def enviar(sesion, url, lector, args, contador, lock):
    """Un hilo emisor. Hay varios en paralelo: cada envío espera la ida y vuelta al servidor
    (~0,3 s), y de a uno no pasan de 3 cuadros/s aunque la conexión dé para más."""
    while True:
        t = time.monotonic()
        img = lector.cuadro
        with lock:  # cada cuadro lo manda un solo hilo
            nuevo = img is not None and img is not contador["ultimo"]
            if nuevo:
                contador["ultimo"] = img
                contador["n"] += 1
                numero = contador["n"]
        if nuevo:
            cuerpo = jpeg(img, args.ancho)
            try:
                r = sesion.post(url, data=cuerpo, headers={"X-Cuadro": str(numero)}, timeout=10)
                if r.status_code == 401:
                    print("[x] El servidor rechazó el ID o el token de la cámara", flush=True)
                    os._exit(1)
                if r.ok:
                    with lock:
                        contador["enviados"] += 1
                else:
                    print(f"[!] El servidor respondió {r.status_code}: {r.text[:200]}", flush=True)
            except requests.RequestException as e:
                print(f"[!] No se pudo enviar: {e}", flush=True)
                time.sleep(2)
        time.sleep(max(0.0, args.hilos / args.fps - (time.monotonic() - t)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camara", required=True, help="URL RTSP/HTTP de la cámara en la red local")
    ap.add_argument("--servidor", required=True, help="ej. https://dev.racvision.org")
    ap.add_argument("--id", required=True, help="ID de la cámara en el panel")
    ap.add_argument("--token", default=os.environ.get("RAC_TOKEN"), help="token de la cámara (o RAC_TOKEN)")
    ap.add_argument("--fps", type=float, default=8, help="cuadros por segundo al servidor (8: video fluido; "
                    "el servidor analiza 2 por segundo)")
    ap.add_argument("--ancho", type=int, default=960, help="ancho máximo del cuadro (960)")
    ap.add_argument("--hilos", type=int, default=4, help="envíos en paralelo (4)")
    a = ap.parse_args()
    if not a.token:
        ap.error("falta el token (--token o RAC_TOKEN)")

    url = f"{a.servidor.rstrip('/')}/admin/api/cameras/{a.id}/cuadro"
    lector = Lector(a.camara)
    lector.start()
    contador, lock = {"n": 0, "enviados": 0, "ultimo": None}, threading.Lock()
    for _ in range(a.hilos):
        sesion = requests.Session()  # una conexión por hilo: Session no es segura entre hilos
        sesion.headers.update({"Authorization": f"Bearer {a.token}", "Content-Type": "image/jpeg"})
        threading.Thread(target=enviar, args=(sesion, url, lector, a, contador, lock), daemon=True).start()
    print(f"[+] Reenviando {a.camara} a {a.servidor} a {a.fps} cuadros/s", flush=True)
    while True:
        time.sleep(60)
        with lock:
            print(f"[+] {contador['enviados']} cuadros enviados en el último minuto", flush=True)
            contador["enviados"] = 0


if __name__ == "__main__":
    main()
