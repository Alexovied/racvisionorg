"""Mide un detector de cascos (.onnx) en ds_casco/test con el mismo código que producción
(helmet_detector.correr_yolo, OpenVINO). Uso: eval_casco.py modelo.onnx [modelo2.onnx ...] [--int8]"""
import glob, json, os, sys, threading, time
from collections import Counter, defaultdict
import cv2, numpy as np

sys.path.insert(0, "/home/user/racvisionorg")
import helmet_detector as hd  # noqa: E402

TEST = "/home/user/work/ds_casco/test"
INT8 = "--int8" in sys.argv
modelos = [a for a in sys.argv[1:] if not a.startswith("--")]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def grupo(nombre):
    for g in ("oi_", "ppe_", "alex_"):
        if nombre.startswith(g):
            if g == "oi_":
                return "openimages_sombreros" if nombre.endswith("_s") else "openimages_cabezas"
            if g == "alex_":
                return "fotos_alex"
            return "construction_ppe"
    return "otro"


def evaluar(ruta):
    net, motor = hd.cargar_modelo(ruta, int8=INT8)
    lock = threading.Lock()
    tab = defaultdict(Counter)
    t = []
    for img_p in sorted(glob.glob(f"{TEST}/images/*.jpg")):
        n = os.path.splitext(os.path.basename(img_p))[0]
        g = grupo(n)
        img = cv2.imread(img_p)
        h, w = img.shape[:2]
        gt = []
        for l in open(f"{TEST}/labels/{n}.txt").read().split("\n"):
            if l.strip():
                c, x, y, bw, bh = map(float, l.split())
                gt.append((int(c), [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
        t0 = time.perf_counter()
        dets = hd.correr_yolo(net, lock, img, 2)
        t.append(time.perf_counter() - t0)
        preds = [(int(sc.argmax()), [b[0], b[1], b[0] + b[2], b[1] + b[3]]) for b, sc in dets]
        if g == "fotos_alex":   # como producción en primer plano: la cabeza más grande
            preds = sorted(preds, key=lambda p: -(p[1][2] - p[1][0]) * (p[1][3] - p[1][1]))[:1]
            real = gt[0][0]
            pred = preds[0][0] if preds else None
            tab[g][("casco" if real == 0 else "sin", "casco" if pred == 0 else "sin" if pred == 1 else "nada")] += 1
            continue
        usado = set()
        for c, b in gt:
            best, bi = 0.3, None
            for i, (pc, pb) in enumerate(preds):
                if i not in usado and iou(b, pb) > best:
                    best, bi = iou(b, pb), i
            real = "casco" if c == 0 else "sin"
            if bi is None:
                tab[g][(real, "nada")] += 1
            else:
                usado.add(bi)
                tab[g][(real, "casco" if preds[bi][0] == 0 else "sin")] += 1
        tab[g][("fondo", "extra")] += len(preds) - len(usado)
    return motor, tab, t


def resumen(tab):
    r = {}
    for g, c in sorted(tab.items()):
        nc = sum(v for (a, b), v in c.items() if a == "casco")
        ns = sum(v for (a, b), v in c.items() if a == "sin")
        r[g] = {
            "casco_ok": f"{c[('casco', 'casco')]}/{nc}", "sin_ok": f"{c[('sin', 'sin')]}/{ns}",
            "sin_visto_casco": c[("sin", "casco")], "casco_visto_sin": c[("casco", "sin")],
            "no_vistas": c[("casco", "nada")] + c[("sin", "nada")], "extras": c[("fondo", "extra")],
        }
    return r


for m in modelos:
    motor, tab, t = evaluar(m)
    print(f"\n=== {m}  ({motor}, {1000 * np.mean(t):.0f} ms/imagen)")
    for g, r in resumen(tab).items():
        print(f"  {g:22s} " + "  ".join(f"{k}={v}" for k, v in r.items()))
    json.dump({g: {f"{a}->{b}": v for (a, b), v in c.items()} for g, c in tab.items()},
              open(m.replace(".onnx", "") + ("_int8" if INT8 else "") + "_eval.json", "w"), indent=1)
