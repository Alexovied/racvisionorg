"""Latencia media por frame de los YOLO de casco y chaleco: cv2.dnn vs OpenVINO (FP32 e INT8, CPU y GPU Intel).

Uso: python benchmark_inferencia.py [carpeta_de_imagenes] [n_imagenes]
Mide correr_yolo completo (letterbox + inferencia + NMS), como en producción.
"""
import glob
import os
import platform
import sys
import time

import cv2
import numpy as np

import helmet_detector

RAIZ = os.path.dirname(os.path.abspath(__file__))
MODELOS = {"casco": os.path.join(RAIZ, "detector_casco.onnx"),
           "chaleco": os.path.join(RAIZ, "detector_chaleco.onnx")}


def imagenes(carpeta, n):
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.jpg")))[:n] or [os.path.join(RAIZ, "test.jpg")]
    imgs = [img for img in map(cv2.imread, rutas) if img is not None]
    assert imgs, f"no hay imágenes en {carpeta}"
    return imgs


def medir(net, imgs, por_clase, vueltas=3):
    lock = helmet_detector._lock
    helmet_detector.correr_yolo(net, lock, imgs[0], 2, por_clase=por_clase)  # calentamiento
    t = []
    for _ in range(vueltas):
        for img in imgs:
            t0 = time.perf_counter()
            helmet_detector.correr_yolo(net, lock, img, 2, por_clase=por_clase)
            t.append(time.perf_counter() - t0)
    return 1000 * np.mean(t), 1000 * np.median(t)


def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "con_casco")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    imgs = imagenes(carpeta, n)
    cpu = helmet_detector._core.get_property("CPU", "FULL_DEVICE_NAME") if helmet_detector._core else platform.processor()
    print(f"CPU: {cpu} | {len(imgs)} imágenes x 3 vueltas | OpenCV {cv2.__version__}")
    devices = [d for d in ("CPU", "GPU") if d in helmet_detector._core.available_devices]
    for d in devices[1:]:
        print(f"{d}: {helmet_detector._core.get_property(d, 'FULL_DEVICE_NAME')}")
    for nombre, ruta in MODELOS.items():
        por_clase = nombre == "chaleco"
        base = medir(cv2.dnn.readNetFromONNX(ruta), imgs, por_clase)
        print(f"{nombre:8s} cv2.dnn          {base[0]:6.1f} ms (mediana {base[1]:.1f})")
        for device in devices:
            for int8 in (False, True):
                net, motor = helmet_detector.cargar_modelo(ruta, int8=int8, device=device)
                assert motor.startswith("OpenVINO"), "OpenVINO no está disponible"
                ms = medir(net, imgs, por_clase)
                print(f"{nombre:8s} {motor:13s} {device} {ms[0]:6.1f} ms (mediana {ms[1]:.1f}) | {base[0] / ms[0]:.2f}x")

if __name__ == "__main__":
    main()
