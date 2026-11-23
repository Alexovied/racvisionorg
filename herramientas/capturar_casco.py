"""Junta fotos de una cámara para reentrenar el detector de casco (entrenamiento/detector_casco_colab.ipynb).

Guarda un cuadro cada --cada segundos en datasets/<clase>/, durante --segundos. Toda la tanda
lleva la misma etiqueta, así que en cada tanda tiene que haber una sola persona a la vista y
siempre en la misma situación (con casco o sin casco). Los cuadros sin nadie no molestan:
el notebook descarta los que no tienen cabeza.

    python herramientas/capturar_casco.py --clase sin_casco --segundos 90
    python herramientas/capturar_casco.py --zip      # arma entrenamiento/fotos_rac.zip para Colab
"""
import argparse
import os
import time
import zipfile

import cv2

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASES = ("con_casco", "sin_casco")


def capturar(camara, clase, segundos, cada, ancho):
    carpeta = os.path.join(RAIZ, "datasets", clase)
    os.makedirs(carpeta, exist_ok=True)
    cap = cv2.VideoCapture(camara, cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000])
    guardadas, t_fin, t_prox = 0, time.monotonic() + segundos, 0.0
    while time.monotonic() < t_fin:
        ok, img = cap.read()  # se lee siempre, para no guardar cuadros viejos del búfer
        if not ok:
            print("[!] Se cortó la cámara, reconecto...", flush=True)
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(camara, cv2.CAP_FFMPEG)
            continue
        if time.monotonic() < t_prox:
            continue
        t_prox = time.monotonic() + cada
        if img.shape[1] > ancho:
            img = cv2.resize(img, (ancho, round(img.shape[0] * ancho / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(carpeta, f"oryx_{time.strftime('%Y%m%d_%H%M%S')}_{guardadas}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        guardadas += 1
        print(f"\r[+] {guardadas} fotos en datasets/{clase}", end="", flush=True)
    cap.release()
    print()


def armar_zip():
    destino = os.path.join(RAIZ, "entrenamiento", "fotos_rac.zip")
    n = 0
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_STORED) as z:  # JPEG ya viene comprimido
        for clase in CLASES:
            for nombre in sorted(os.listdir(os.path.join(RAIZ, "datasets", clase))):
                if nombre.lower().endswith(".jpg"):
                    z.write(os.path.join(RAIZ, "datasets", clase, nombre), f"{clase}/{nombre}")
                    n += 1
    print(f"[+] {destino}: {n} fotos")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camara", default="rtsp://10.10.10.124:554/onvif1")
    ap.add_argument("--clase", choices=CLASES)
    ap.add_argument("--segundos", type=float, default=90)
    ap.add_argument("--cada", type=float, default=1.0, help="segundos entre fotos (1)")
    ap.add_argument("--ancho", type=int, default=960)
    ap.add_argument("--zip", action="store_true", help="armar el zip para Colab con todas las fotos")
    a = ap.parse_args()
    if a.clase:
        capturar(a.camara, a.clase, a.segundos, a.cada, a.ancho)
    if a.zip:
        armar_zip()
    if not (a.clase or a.zip):
        ap.error("indicá --clase y/o --zip")
