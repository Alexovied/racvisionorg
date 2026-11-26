"""Arma el dataset YOLO de cabezas (0 casco / 1 sin_casco) con train / test separados.

Fuentes:
- Construction-PPE (Ultralytics, CC BY 4.0): sus cajas 'helmet' son solo el casco, así que la caja de
  la cabeza sale del modelo actual (pseudo-etiqueta) y la CLASE sale de la etiqueta humana
  (helmet / no_helmet). train = su train, test = su val.
- Open Images V7 (CC BY 4.0, imágenes CC BY 2.0): cabezas humanas sin cascos; si hay un sombrero/gorra
  encima, la caja se extiende hacia arriba para incluirlo. Todas sin_casco. 75 % train, 25 % test por imagen.
- Fotos de Alex (con_casco/ sin_casco/): como el notebook, la cabeza más grande y la clase de la carpeta.
  70 % train / 30 % test con random.seed(1) (el mismo corte que evaluar_modelos.py).
- 100 cuadros de Pexels etiquetados a mano (rac_auto/etiquetas.json): solo train (el modelo actual ya los vio).
"""
import glob, hashlib, json, os, random, shutil, sys
from collections import Counter
import cv2
from ultralytics import YOLO

W = "/home/user/work"
OUT = f"{W}/ds_casco"
BASE_PT = "/mnt/project-files/modelo-casco/cerca_20261125/detector_casco.pt"
modelo = YOLO(BASE_PT)
cont = Counter()


def escribir(split, nombre, img, cajas, origen):
    """cajas: [(clase, x0, y0, x1, y1)] en píxeles."""
    h, w = img.shape[:2]
    os.makedirs(f"{OUT}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUT}/{split}/labels", exist_ok=True)
    cv2.imwrite(f"{OUT}/{split}/images/{nombre}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lineas = []
    for c, x0, y0, x1, y1 in cajas:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        lineas.append(f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        cont[(split, origen, c)] += 1
    open(f"{OUT}/{split}/labels/{nombre}.txt", "w").write("\n".join(lineas))
    cont[(split, origen, "img")] += 1


def predecir(img, conf=0.2):
    r = modelo.predict(img, conf=conf, imgsz=640, verbose=False)[0]
    return [(int(c), float(p), *map(float, b)) for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]


# ── Construction-PPE ─────────────────────────────────────────────────────────
PPE = f"{W}/dl/ppe/construction-ppe"
for split_src, split in (("train", "train"), ("val", "test")):
    for ruta in sorted(glob.glob(f"{PPE}/images/{split_src}/*")):
        img = cv2.imread(ruta)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = []
        for l in open(f"{PPE}/labels/{split_src}/" + os.path.splitext(os.path.basename(ruta))[0] + ".txt"):
            c, x, y, bw, bh = map(float, l.split())
            if int(c) in (0, 6):
                gt.append((0 if int(c) == 0 else 1, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
        preds = predecir(img)
        usadas, cajas = set(), []
        for c, x0, y0, x1, y1 in gt:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            # la cabeza del modelo que contiene el centro de la caja humana (casco: en su mitad de arriba)
            cand = [(i, p) for i, p in enumerate(preds) if i not in usadas
                    and p[2] <= cx <= p[4] and p[3] <= cy <= p[5]
                    and (c == 1 or cy <= p[3] + 0.6 * (p[5] - p[3]))
                    and 0.4 < (p[4] - p[2]) / max(1, x1 - x0) < 2.5]
            if cand:
                i, p = max(cand, key=lambda t: t[1][1])
                usadas.add(i)
                cajas.append((c, p[2], p[3], p[4], p[5]))
            elif c == 1:
                cajas.append((1, x0, y0, x1, y1))           # no_helmet ya es la cabeza
            else:                                            # casco solo: se estira hacia abajo hasta la cara
                hh = y1 - y0
                cajas.append((0, x0, y0, x1, y1 + 0.9 * hh))
        # cabezas que el modelo ve con seguridad y no tienen etiqueta humana: se dejan con la clase del modelo
        for i, p in enumerate(preds):
            if i not in usadas and p[1] >= 0.6:
                cajas.append((p[0], p[2], p[3], p[4], p[5]))
        escribir(split, "ppe_" + os.path.splitext(os.path.basename(ruta))[0][-16:], img, cajas, "ppe")

# ── Open Images ──────────────────────────────────────────────────────────────
for s in json.load(open(f"{W}/oi/seleccion.json")):
    img = cv2.imread(f"{W}/oi/img/{s['id']}.jpg")
    if img is None:
        continue
    h, w = img.shape[:2]
    split = "test" if int(hashlib.md5(s["id"].encode()).hexdigest(), 16) % 4 == 0 else "train"
    cajas = []
    for x0, y0, x1, y1 in s["heads"]:
        x0, y0, x1, y1 = x0 * w, y0 * h, x1 * w, y1 * h
        hw = x1 - x0
        for a0, b0, a1, b1 in s["hats"]:
            a0, b0, a1, b1 = a0 * w, b0 * h, a1 * w, b1 * h
            inter = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
            if inter > 0.3 * (a1 - a0) * (b1 - b0) or (x0 <= (a0 + a1) / 2 <= x1 and b1 >= y0 and b0 <= y0 + 0.5 * (y1 - y0)):
                y0 = min(y0, b0)
                cx = (x0 + x1) / 2
                x0, x1 = max(min(x0, a0), cx - 0.8 * hw), min(max(x1, a1), cx + 0.8 * hw)
        cajas.append((1, x0, y0, x1, y1))
    escribir(split, "oi_" + s["id"] + ("_s" if s["hats"] else ""), img, cajas, "oi_sombrero" if s["hats"] else "oi")

# ── Fotos de Alex ────────────────────────────────────────────────────────────
D = "/mnt/user-data/uploads/rac/datasets"
fotos = [(p, real) for real in ("con_casco", "sin_casco") for p in sorted(glob.glob(f"{D}/{real}/*.jpg"))]
random.seed(1)
random.shuffle(fotos)
corte = int(len(fotos) * 0.7)
for k, (p, real) in enumerate(fotos):
    split = "train" if k < corte else "test"
    img = cv2.imread(p)
    preds = predecir(img, 0.15)
    c = 0 if real == "con_casco" else 1
    nombre = f"alex_{real}_{os.path.splitext(os.path.basename(p))[0]}"
    if not preds:
        cont[(split, "alex", "sin_caja")] += 1
        if split == "test":   # en test igual se guarda (sin caja = la imagen entera es la cabeza)
            h, w = img.shape[:2]
            escribir(split, nombre, img, [(c, 0, 0, w, h)], "alex")
        continue
    mejor = max(preds, key=lambda q: (q[4] - q[2]) * (q[5] - q[3]))
    escribir(split, nombre, img, [(c, *mejor[2:])], "alex")

# ── Cuadros de Pexels etiquetados a mano ─────────────────────────────────────
et = json.load(open(f"{W}/rac_auto/etiquetas.json"))["cajas"]
for f, cs in et.items():
    img = cv2.imread(f"{W}/rac_pexels/cuadros/{f}")
    if img is None:
        continue
    h, w = img.shape[:2]
    cajas = [(0 if b["casco"] == "con" else 1, b["bbox"][0] * w, b["bbox"][1] * h,
              (b["bbox"][0] + b["bbox"][2]) * w, (b["bbox"][1] + b["bbox"][3]) * h)
             for b in cs if b["casco"] in ("con", "sin")]
    if any(b["casco"] is None for b in cs):
        continue   # cabeza dudosa sin clase: mejor no usar el cuadro
    escribir("train", "pexels_" + os.path.splitext(f)[0], img, cajas, "pexels")

for k in sorted(cont, key=str):
    print(k, cont[k])
