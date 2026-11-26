"""Mide las reglas del hombre caído (las de pose.py / motor.acostados) sobre un volcado de pose de GMDCSA24.
Uso: eval_caidas.py gm_pose.pkl [modelo] [sujetos=1,2,3,4]"""
import json, math, pickle, sys
from collections import Counter
import numpy as np

HOMBROS, CADERAS = (5, 6), (11, 12)
PR16 = dict(conf_persona=0.25, conf_punto=0.3, angulo=40, cabeza=True, caja=True, ratio=1.3)
SIN_CAIDA = {"Sitting", "Walking", "Standing", "Reading", "Drinking", "Writing", "Eating", "Exercising"}


def torso(kp, p):
    ok = lambda idx: min(kp[i][2] for i in idx) >= p["conf_punto"]
    if not ok(HOMBROS):
        return None
    if ok(CADERAS):
        a = np.mean([kp[i][:2] for i in HOMBROS], 0); b = np.mean([kp[i][:2] for i in CADERAS], 0)
    elif p["cabeza"] and kp[0][2] >= p["conf_punto"]:
        a = kp[0][:2]; b = np.mean([kp[i][:2] for i in HOMBROS], 0)
    else:
        return None
    ang = math.degrees(math.atan2(a[0] - b[0], b[1] - a[1]))
    return abs((ang + 180) % 360 - 180) > p["angulo"]


def caja_ancha(b, forma, ratio):
    x, y, w, h = b; alto, ancho = forma
    return not (w < ratio * h or x < 4 or y < 4 or x + w > ancho - 4 or y + h > alto - 4)


def acostado(fr, modelo, p, extra=None):
    for b, conf, kp in fr[modelo]:
        if conf < p["conf_persona"]:
            continue
        t = torso(kp, p)
        if extra:
            t = extra(b, conf, kp, fr, t, p)
        if t or (t is None and p["caja"] and caja_ancha(b, fr["forma"], p["ratio"])):
            return True
    return False


def etiqueta(info, t):
    """'piso' (tiene que verse acostado), 'derecho' (no), 'cama' (acostado a propósito) o None (dudoso)."""
    estados = set()
    for nom, a, b in info["segs"]:
        if not a <= t <= b:
            continue
        if nom.startswith("Fall"):
            estados.add("piso" if t >= b - max(0.4, 0.4 * (b - a)) else None)
        elif nom == "Sleeping" or (nom == "Exercising" and "push" in info["desc"].lower()):
            estados.add("cama")
        elif nom in SIN_CAIDA:
            estados.add("derecho")
    estados.discard(None)
    return estados.pop() if len(estados) == 1 else None


def evaluar(datos, clips, modelo, p, extra=None, sujetos=None):
    fr_c = Counter(); clip_c = Counter(); detalle = {}
    for nom, cuadros in datos.items():
        if sujetos and int(nom[1]) not in sujetos:
            continue
        info = clips[nom]
        racha, peor = 0, 0
        piso = []
        for fr in cuadros:
            e = etiqueta(info, fr["t"])
            a = acostado(fr, modelo, p, extra)
            if e:
                fr_c[(e, a)] += 1
            if e == "piso":
                piso.append(a)
            racha = racha + 1 if (a and e == "derecho") else 0
            peor = max(peor, racha)
        if "_Fall_" in nom:
            visto = bool(piso) and np.mean(piso) >= 0.5
            clip_c[("caida", visto)] += 1
            detalle[nom] = round(float(np.mean(piso)), 2) if piso else None
        elif any(s[0] == "Sleeping" for s in info["segs"]) or "push" in info["desc"].lower():
            pass
        else:
            clip_c[("adl", peor >= 5)] += 1      # 1 s seguido "acostado" estando derecho = falsa alarma
            if peor >= 5:
                detalle[nom] = f"falsa ({peor} cuadros)"
    return fr_c, clip_c, detalle


def informe(fr_c, clip_c):
    piso = fr_c[("piso", True)] / max(1, fr_c[("piso", True)] + fr_c[("piso", False)])
    fp = fr_c[("derecho", True)] / max(1, fr_c[("derecho", True)] + fr_c[("derecho", False)])
    cama = fr_c[("cama", True)] / max(1, fr_c[("cama", True)] + fr_c[("cama", False)])
    nc = clip_c[("caida", True)] + clip_c[("caida", False)]; na = clip_c[("adl", True)] + clip_c[("adl", False)]
    return (f"caídas vistas {clip_c[('caida', True)]}/{nc} clips | cuadros en el piso vistos {100 * piso:.0f}% | "
            f"falsas: {clip_c[('adl', True)]}/{na} clips, {100 * fp:.1f}% de cuadros derechos | acostado en cama {100 * cama:.0f}%")


if __name__ == "__main__":
    datos = pickle.load(open(sys.argv[1], "rb"))
    clips = json.load(open("/home/user/work/gmdcsa.json"))
    modelos = [k for k in next(iter(datos.values()))[0] if k not in ("t", "forma", "cabezas") and not k.startswith("ms_")]
    for m in modelos:
        fr_c, clip_c, det = evaluar(datos, clips, m, PR16)
        print(m, informe(fr_c, clip_c))
        print("  ", {k: v for k, v in det.items() if v is not None and (isinstance(v, str) or v < 0.5)})
