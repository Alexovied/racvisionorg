"""Mide el chaleco con el código de producción (epp.py): modelo + regla de color.
Uso: eval_chaleco.py detector_chaleco.onnx [...] [--int8] [--sin-color]
- construction_ppe (val, 200 fotos): cada persona etiquetada tiene 'vest' o 'no_vest' adentro.
- fotos_alex: personas/ y personas_prueba/ (chaleco_* = con chaleco) por la cabeza, como evaluar_modelos.py.
"""
import glob, os, sys, time
from collections import Counter, defaultdict
import cv2, numpy as np

sys.path.insert(0, "/home/user/racvisionorg")
import helmet_detector as hd  # noqa: E402
import epp  # noqa: E402

INT8 = "--int8" in sys.argv
if "--asignar" in sys.argv:
    sys.path.insert(0, "/home/user/work/scripts"); import epp_asignar; epp.detectar_personas = epp_asignar.detectar_personas
for a in sys.argv:
    if a.startswith("--conf-chaleco="):
        _t = float(a.split("=")[1]); _orig = epp.detectar
        epp.detectar = lambda img: [(b, s) for b, s in _orig(img) if s.argmax() == 0 or s.max() >= _t]
if "--sin-color" in sys.argv:
    epp.proporcion_alta_visibilidad = lambda region: 0.0
PPE = "/home/user/work/dl/ppe/construction-ppe"
ALEX = "/mnt/user-data/uploads/rac/datasets"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def dentro(c, p):
    return p[0] <= c[0] <= p[2] and p[1] <= c[1] <= p[3]


def gt_ppe(ruta, img):
    h, w = img.shape[:2]
    cajas = defaultdict(list)
    for l in open(ruta.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"):
        c, x, y, bw, bh = map(float, l.split())
        cajas[int(c)].append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    personas = []
    for p in cajas[5]:
        con = any(dentro(((v[0] + v[2]) / 2, (v[1] + v[3]) / 2), p) for v in cajas[2])
        sin = any(dentro(((v[0] + v[2]) / 2, (v[1] + v[3]) / 2), p) for v in cajas[11])
        personas.append((p, "chaleco" if con and not sin else "sin" if sin and not con else None))
    return personas


def evaluar(ruta_modelo):
    epp._net, motor = hd.cargar_modelo(ruta_modelo, int8=INT8)
    tab = defaultdict(Counter)
    t = []
    for ruta in sorted(glob.glob(f"{PPE}/images/val/*")):
        img = cv2.imread(ruta)
        t0 = time.perf_counter()
        preds = epp.detectar_personas(img)
        t.append(time.perf_counter() - t0)
        pb = [([b[0], b[1], b[0] + b[2], b[1] + b[3]], v) for b, v in ((p["bbox"], p["chaleco"]) for p in preds)]
        usado = set()
        for g, real in gt_ppe(ruta, img):
            if real is None:
                continue
            best, bi = 0.4, None
            for i, (b, v) in enumerate(pb):
                if i not in usado and iou(g, b) > best:
                    best, bi = iou(g, b), i
            if bi is None:
                tab["construction_ppe"][(real, "no vista")] += 1
            else:
                usado.add(bi)
                tab["construction_ppe"][(real, "chaleco" if pb[bi][1] else "sin")] += 1
    rutas = sorted(glob.glob(f"{ALEX}/personas/*/*.jpg") + glob.glob(f"{ALEX}/personas_prueba/*/*.jpg"))
    for ruta in rutas:
        img = cv2.imread(ruta)
        real = "chaleco" if os.path.basename(ruta).startswith("chaleco_") else "sin"
        cabezas = hd.detectar(img)
        if not cabezas:
            tab["fotos_alex"][(real, "sin cabeza")] += 1
            continue
        v = epp.chalecos_por_cabeza(img, [cabezas[0]["bbox"]])[0]
        tab["fotos_alex"][(real, {True: "chaleco", False: "sin", None: "torso no visible"}[v])] += 1
    return motor, tab, t


for m in [a for a in sys.argv[1:] if not a.startswith("--")]:
    motor, tab, t = evaluar(m)
    print(f"\n=== {m} ({motor}, {1000 * np.mean(t):.0f} ms/imagen PPE)")
    for g, c in tab.items():
        nc = sum(v for (a, b), v in c.items() if a == "chaleco"); ns = sum(v for (a, b), v in c.items() if a == "sin")
        print(f"  {g:18s} chaleco_ok={c[('chaleco', 'chaleco')]}/{nc}  sin_ok={c[('sin', 'sin')]}/{ns}  "
              f"sin_visto_con={c[('sin', 'chaleco')]}  con_visto_sin={c[('chaleco', 'sin')]}  "
              + "  ".join(f"{a}->{b}={v}" for (a, b), v in sorted(c.items()) if b not in ("chaleco", "sin")))
