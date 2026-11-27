"""Dataset YOLO 0 persona / 1 chaleco.
- Construction-PPE (CC BY 4.0): Person -> persona, vest -> chaleco. train = su train; su val queda para la prueba.
- Open Images (CC BY): personas (Person/Man/Woman/Boy/Girl) de las imágenes ya bajadas, del lado 'train'
  del mismo corte que el casco. Sin chalecos (gente común); imágenes con cajas de grupo se descartan."""
import csv, collections, glob, hashlib, json, os, random, shutil
import cv2
W = "/home/user/work"; OUT = f"{W}/ds_chaleco"
PPE = f"{W}/dl/ppe/construction-ppe"
for s in ("train", "val"):
    os.makedirs(f"{OUT}/{s}/images", exist_ok=True); os.makedirs(f"{OUT}/{s}/labels", exist_ok=True)
cont = collections.Counter()
ppe = sorted(glob.glob(f"{PPE}/images/train/*"))
random.seed(7); random.shuffle(ppe)
for k, ruta in enumerate(ppe):
    s = "val" if k < 80 else "train"
    n = os.path.splitext(os.path.basename(ruta))[0]
    lin = []
    for l in open(f"{PPE}/labels/train/{n}.txt"):
        p = l.split()
        if p and int(p[0]) in (5, 2):
            lin.append(" ".join([("0" if p[0] == "5" else "1")] + p[1:])); cont[(s, "ppe", p[0])] += 1
    shutil.copy(ruta, f"{OUT}/{s}/images/ppe_{n}{os.path.splitext(ruta)[1]}")
    open(f"{OUT}/{s}/labels/ppe_{n}.txt", "w").write("\n".join(lin))
PERS = {"/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"}
sel = {s["id"] for s in json.load(open(f"{W}/oi/seleccion.json"))}
cajas = collections.defaultdict(list); grupo = set()
for f in ("validation-annotations-bbox.csv", "test-annotations-bbox.csv"):
    for r in csv.DictReader(open(f"{W}/oi/{f}")):
        if r["ImageID"] in sel and r["LabelName"] in PERS:
            if r["IsGroupOf"] == "1" or r["IsDepiction"] == "1":
                grupo.add(r["ImageID"])
            cajas[r["ImageID"]].append([float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax")])
n_oi = 0
for i, cs in cajas.items():
    if i in grupo or int(hashlib.md5(i.encode()).hexdigest(), 16) % 4 == 0:
        continue
    # Man/Woman/Person repiten a la misma persona: se sacan duplicados
    uniq = []
    for c in cs:
        if not any(max(0, min(c[2], u[2]) - max(c[0], u[0])) * max(0, min(c[3], u[3]) - max(c[1], u[1])) >
                   0.8 * min((c[2] - c[0]) * (c[3] - c[1]), (u[2] - u[0]) * (u[3] - u[1])) for u in uniq):
            uniq.append(c)
    s = "val" if n_oi % 12 == 0 else "train"; n_oi += 1
    shutil.copy(f"{W}/oi/img/{i}.jpg", f"{OUT}/{s}/images/oi_{i}.jpg")
    open(f"{OUT}/{s}/labels/oi_{i}.txt", "w").write("\n".join(
        f"0 {(c[0]+c[2])/2:.6f} {(c[1]+c[3])/2:.6f} {c[2]-c[0]:.6f} {c[3]-c[1]:.6f}" for c in uniq))
    cont[(s, "oi", "persona")] += len(uniq)
open(f"{OUT}/data.yaml", "w").write(f"path: {OUT}\ntrain: train/images\nval: val/images\nnames:\n  0: persona\n  1: chaleco\n")
for k in sorted(cont): print(k, cont[k])
print(len(os.listdir(f"{OUT}/train/images")), len(os.listdir(f"{OUT}/val/images")))
