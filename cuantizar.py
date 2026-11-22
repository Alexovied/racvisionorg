"""Cuantiza los YOLO de casco y chaleco a INT8 con NNCF (Intel) para OpenVINO.

Uso: python cuantizar.py   (necesita `pip install nncf` y las fotos de con_casco/ y sin_casco/)
Genera detector_casco_int8.xml/.bin y detector_chaleco_int8.xml/.bin, que
helmet_detector.cargar_modelo usa si existen. Calibra con el 70 % de las fotos
que el notebook usó para entrenar: el 30 % de prueba de evaluar_modelos.py no se toca.
"""
import os

import cv2
import nncf
import openvino as ov

import evaluar_modelos
import helmet_detector

RAIZ = os.path.dirname(os.path.abspath(__file__))


def fotos_calibracion():
    prueba = {r for r, _ in evaluar_modelos.fotos_prueba_casco()}
    rutas = [r for real in ("con_casco", "sin_casco")
             for r in sorted(evaluar_modelos.glob.glob(os.path.join(RAIZ, real, "*.jpg"))) if r not in prueba]
    assert len(rutas) >= 50, "faltan fotos de calibración en con_casco/ y sin_casco/"
    return rutas


def preparar(ruta):
    # El mismo preprocesamiento que correr_yolo en producción
    canvas = helmet_detector._letterbox(cv2.imread(ruta))[0]
    return cv2.dnn.blobFromImage(canvas, 1 / 255.0, (helmet_detector.INPUT_SIZE,) * 2, swapRB=True)


def main():
    rutas = fotos_calibracion()
    print(f"Calibración con {len(rutas)} fotos")
    for nombre in ("casco", "chaleco"):
        modelo = ov.Core().read_model(os.path.join(RAIZ, f"detector_{nombre}.onnx"))
        int8 = nncf.quantize(modelo, nncf.Dataset(rutas, preparar), preset=nncf.QuantizationPreset.MIXED,
                             subset_size=len(rutas),
                             # La cabeza de YOLOv8 (decodificación de cajas) pierde precisión en INT8
                             ignored_scope=nncf.IgnoredScope(types=["Multiply", "Subtract", "Sigmoid"]))
        ov.save_model(int8, os.path.join(RAIZ, f"detector_{nombre}_int8.xml"))
        print(f"[+] detector_{nombre}_int8.xml")


if __name__ == "__main__":
    main()
