"""Reglas del hombre caído v4: pose derecha + pose con la imagen girada 90°, y descarte de "agachado"
(piernas verticales debajo de la cadera) y de esqueletos más altos que anchos.
Uso: reglas2.py gm_rot.pkl [px_rot.pkl]"""
import itertools, json, math, pickle, sys
from collections import Counter, deque
import numpy as np
sys.path.insert(0, "/home/user/work/scripts")
from eval_caidas import etiqueta, informe, caja_ancha

HOMBROS, CADERAS, RODILLAS, TOBILLOS = (5, 6), (11, 12), (13, 14), (15, 16)


def angulo_vert(a, b):
    """grados entre el vector a->b (a arriba) y la vertical hacia abajo"""
    ang = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))
    return abs((ang + 180) % 360 - 180)


def decide(kp, b, forma, p):
    cp = p["conf_punto"]
    vis = lambda idx: min(kp[i][2] for i in idx) >= cp
    if not vis(HOMBROS):
        t = None
    else:
        hm = np.mean([kp[i][:2] for i in HOMBROS], 0)
        if vis(CADERAS):
            cd = np.mean([kp[i][:2] for i in CADERAS], 0)
            t = angulo_vert(hm, cd) > p["angulo"]
        elif kp[0][2] >= cp:
            t = angulo_vert(kp[0][:2], hm) > p["angulo"]
        else:
            t = None
    if t and p["piernas"]:
        # agachado / sentado inclinado: cadera arriba y tobillos (o rodillas) casi en vertical debajo
        for pie in (TOBILLOS, RODILLAS):
            if vis(CADERAS) and vis(pie):
                cd = np.mean([kp[i][:2] for i in CADERAS], 0); pp = np.mean([kp[i][:2] for i in pie], 0)
                largo = np.linalg.norm(pp - cd)
                if angulo_vert(cd, pp) < p["piernas"] and largo > p.get("largo", 0.3) * np.linalg.norm(hm - cd):
                    t = False
                break
    if t and p["aspecto"]:
        pts = np.array([q[:2] for q in kp if q[2] >= cp])
        if len(pts) >= 5:
            w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
            if w < p["aspecto"] * h:
                t = False
    return t


def acostado(fr, p):
    fuentes = ["pose"] + (["cw", "ccw"] if p["girar"] else [])
    for f in fuentes:
        for b, conf, kp in fr[f]:
            if conf < (p["conf_persona"] if f == "pose" else p["conf_girada"]):
                continue
            t = decide(kp, b, fr["forma"], p)
            if t or (f == "pose" and t is None and p["caja"] and caja_ancha(b, fr["forma"], 1.3)):
                return True
    return False


def eval_gm(datos, clips, p, sujetos):
    fr_c = Counter(); clip_c = Counter()
    for nom, cuadros in datos.items():
        if int(nom[1]) not in sujetos:
            continue
        info = clips[nom]; racha = peor = 0; piso = []
        for fr in cuadros:
            e = etiqueta(info, fr["t"]); a = acostado(fr, p)
            if e: fr_c[(e, a)] += 1
            if e == "piso": piso.append(a)
            racha = racha + 1 if (a and e == "derecho") else 0; peor = max(peor, racha)
        if "_Fall_" in nom:
            clip_c[("caida", bool(piso) and np.mean(piso) >= 0.5)] += 1
        elif not (any(s[0] == "Sleeping" for s in info["segs"]) or "push" in info["desc"].lower()):
            clip_c[("adl", peor >= 5)] += 1
    return fr_c, clip_c


ENTRENO = {"10535931": True, "8944315": True, "7925403": True, "7925617": True, "8526604": True, "6520238": True,
           "5434223": False, "8965384": False, "8719392": False, "6290147": False}
VALIDA = {"6909679": True, "7667144": True, "8076896": True, "8077540": True, "9484799": True, "9515572": True,
          "9515578": True, "9830163": True, "10817415": False, "4281405": False, "4292578": False, "5495836": False,
          "6840576": False, "7325946": False, "9484892": False}


def eval_px(datos, p, conj, alerta_s=3.0):
    ok = falsas = 0
    for c, espera in conj.items():
        v = deque(maxlen=8); desde = None; alerta = None
        for fr in datos[c]:
            v.append(acostado(fr, p))
            if sum(v) / len(v) < 0.6: desde = None
            elif desde is None: desde = fr["t"]
            if desde is not None and alerta is None and fr["t"] - desde >= alerta_s: alerta = fr["t"]
        ok += espera and alerta is not None; falsas += (not espera) and alerta is not None
    return ok, falsas


if __name__ == "__main__":
    gm = pickle.load(open(sys.argv[1], "rb")); clips = json.load(open("/home/user/work/gmdcsa.json"))
    px = pickle.load(open(sys.argv[2], "rb")) if len(sys.argv) > 2 else None
    PR16 = dict(conf_persona=0.25, conf_girada=0.5, conf_punto=0.3, angulo=40, caja=True, girar=False, piernas=0, aspecto=0)
    REJ = dict(conf_persona=[0.25], conf_girada=[0.3, 0.4, 0.5, 0.6], conf_punto=[0.3, 0.4], angulo=[40, 50],
               caja=[True, False], girar=[False, True], piernas=[0, 25, 35], aspecto=[0, 0.6, 0.8, 1.0])

    def linea(p):
        s = "ENTRENO s1+s2: " + informe(*eval_gm(gm, clips, p, {1, 2})) + "\n   PRUEBA s3+s4: " + informe(*eval_gm(gm, clips, p, {3, 4}))
        if px:
            a, f = eval_px(px, p, ENTRENO); av, fv = eval_px(px, p, VALIDA)
            s += f"\n   PEXELS (alerta 3 s): entreno {a}/6 caídas, {f}/4 falsas | validación {av}/8, {fv}/7 falsas"
        return s
    print("PR #16:\n  ", linea(PR16))
    res = []
    for vals in itertools.product(*REJ.values()):
        p = dict(zip(REJ, vals))
        if not p["girar"] and p["conf_girada"] != 0.5:
            continue
        fr_c, clip_c = eval_gm(gm, clips, p, {1, 2})
        piso = fr_c[("piso", True)] / max(1, fr_c[("piso", True)] + fr_c[("piso", False)])
        fp = fr_c[("derecho", True)] / max(1, fr_c[("derecho", True)] + fr_c[("derecho", False)])
        puntaje = clip_c[("caida", True)] - 2 * clip_c[("adl", True)] + 5 * piso - 20 * fp
        res.append((puntaje, p))
    res.sort(key=lambda x: -x[0])
    for puntaje, p in res[:6]:
        print(f"\n{puntaje:.1f} {p}\n  ", linea(p))
    pickle.dump(res[:30], open("/home/user/work/reglas2_mejores.pkl", "wb"))
