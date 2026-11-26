"""Pose por cuadro (5/s) en la imagen derecha y girada 90° a cada lado (puntos devueltos a la imagen original).
Uso: volcar_rot.py salida.pkl pose.onnx lista.json   (lista: {nombre: ruta})"""
import os, pickle, sys, threading, json
sys.path.insert(0, "/home/user/racvisionorg")
import cv2, numpy as np
import helmet_detector
sal, mod, lista = sys.argv[1:4]
clips = json.load(open(lista))
red = helmet_detector.cargar_modelo(mod, int8=False)[0]; lock = threading.Lock(); datos = {}
def correr(img):
    return [(list(map(float, b)), float(np.max(s)), np.asarray(p, dtype=np.float32)) for b, s, p in helmet_detector.correr_yolo(red, lock, img, 1, 0.15, puntos=True)]
for nom, ruta in clips.items():
    cap = cv2.VideoCapture(ruta); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    paso = max(1, round(fps / 5)); n = 0; cuadros = []
    while True:
        ok, img = cap.read()
        if not ok: break
        n += 1
        if (n - 1) % paso: continue
        h0, w0 = img.shape[:2]; f = 960 / max(h0, w0); img = cv2.resize(img, (round(w0 * f), round(h0 * f)))
        H, W = img.shape[:2]
        fr = {"t": (n - 1) / fps, "forma": (H, W), "pose": correr(img)}
        for clave, rot in (("cw", cv2.ROTATE_90_CLOCKWISE), ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
            res = []
            for b, c, kp in correr(cv2.rotate(img, rot)):
                x, y, w, h = b; kp = kp.copy()
                if clave == "cw":   # (x', y') -> (y', H-1-x')
                    b2 = [y, H - x - w, h, w]; kx = kp[:, 1].copy(); ky = H - 1 - kp[:, 0]
                else:               # (x', y') -> (W-1-y', x')
                    b2 = [W - y - h, x, h, w]; kx = W - 1 - kp[:, 1]; ky = kp[:, 0].copy()
                kp[:, 0], kp[:, 1] = kx, ky
                res.append((b2, c, kp))
            fr[clave] = res
        cuadros.append(fr)
    datos[nom] = cuadros; print(nom, len(cuadros), flush=True)
pickle.dump(datos, open(sal, "wb"))
