"""Precisión y latencia de casco y chaleco con fotos propias del equipo, con el pipeline de producción.

Uso: python evaluar_modelos.py
- Casco: el mismo 30 % de con_casco/ y sin_casco/ que entrenamiento/detector_casco_colab.ipynb
  dejó afuera del entrenamiento (random.seed(1)), así que son fotos que el modelo nunca vio.
- Chaleco: fotos de personas/ y personas_prueba/ ("chaleco_*" = con chaleco, el resto sin chaleco)
  y las cabezas de sin_casco/ (nadie tiene chaleco) para contar falsos "con chaleco".
Las carpetas de fotos no están en git (datos personales); sin ellas el script avisa y sigue.
"""
import glob
import os
import random
import time
from collections import Counter

import cv2
import numpy as np

import epp
import helmet_detector

RAIZ = os.path.dirname(os.path.abspath(__file__))


def fotos_prueba_casco():
    # Igual que el notebook: lista ordenada por carpeta, seed 1, 70 % ajuste / 30 % prueba
    fotos = [(p, real) for real in ("con_casco", "sin_casco")
             for p in sorted(glob.glob(os.path.join(RAIZ, real, "*.jpg")))]
    random.seed(1)
    random.shuffle(fotos)
    return fotos[int(len(fotos) * 0.7):]


def evaluar_casco():
    tabla, t = Counter(), []
    fotos = fotos_prueba_casco()
    for ruta, real in fotos:
        img = cv2.imread(ruta)
        t0 = time.perf_counter()
        dets = helmet_detector.detectar(img)
        t.append(time.perf_counter() - t0)
        # producción: la cabeza más grande; el detector dice "casco", la carpeta "con_casco"
        pred = {"casco": "con_casco", "sin_casco": "sin_casco"}[dets[0]["clase"]] if dets else "nada"
        tabla[(real, pred)] += 1
    return fotos, tabla, t


def evaluar_chaleco():
    rutas = sorted(glob.glob(os.path.join(RAIZ, "personas", "*", "*.jpg"))
                   + glob.glob(os.path.join(RAIZ, "personas_prueba", "*", "*.jpg")))
    rutas += sorted(glob.glob(os.path.join(RAIZ, "sin_casco", "*.jpg")))
    tabla, t = Counter(), []
    for ruta in rutas:
        img = cv2.imread(ruta)
        real = "chaleco" if os.path.basename(ruta).startswith("chaleco_") else "sin_chaleco"
        cabezas = helmet_detector.detectar(img)
        if not cabezas:
            tabla[(real, "sin cabeza")] += 1
            continue
        t0 = time.perf_counter()
        v = epp.chalecos_por_cabeza(img, [cabezas[0]["bbox"]])[0]
        t.append(time.perf_counter() - t0)
        pred = {True: "chaleco", False: "sin_chaleco", None: "no se ve el torso"}[v]
        tabla[(real, pred)] += 1
    return tabla, t


def imprimir(titulo, tabla, t):
    print(f"\n== {titulo}")
    for (real, pred), n in sorted(tabla.items()):
        print(f"  real={real:12s} predicho={pred:18s} {n}")
    juzgadas = {k: n for k, n in tabla.items() if k[1] in ("con_casco", "sin_casco", "chaleco", "sin_chaleco", "nada")}
    ok = sum(n for (r, p), n in juzgadas.items() if r == p)
    print(f"  aciertos {ok}/{sum(juzgadas.values())} | latencia media {1000 * np.mean(t):.0f} ms")


def main():
    motor = "OpenVINO" if helmet_detector._core is not None else "cv2.dnn"
    print(f"Motor: {motor}")
    if not helmet_detector.disponible():
        print("Falta detector_casco.onnx")
        return
    fotos, tabla, t = evaluar_casco()
    if not fotos:
        print("No están las carpetas con_casco/ y sin_casco/")
    else:
        imprimir(f"Casco: {len(fotos)} fotos que el modelo no vio al entrenar", tabla, t)
    tabla, t = evaluar_chaleco()
    if tabla:
        imprimir("Chaleco (modelo + regla de color), cabeza más grande de cada foto", tabla, t)


if __name__ == "__main__":
    main()
