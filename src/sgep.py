"""SGEP-Dashboard: encuestas para evaluar los proyectos de 7mo (Expotécnica).

Módulo dentro del panel de RAC Vision: reutiliza el login y los roles de admin.py,
la protección CSRF de app.py, admin.css y data_store (JSON en data/sgep_*.json).

Roles: admin (todo), pm (gestiona proyectos, encuestas y envíos),
auditor (solo lectura: dashboard, listados y exportaciones).
"""
import io
import json
import os
import re
import secrets
import smtplib
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from email.message import EmailMessage

from flask import (
    Blueprint, abort, flash, jsonify, redirect, render_template, request,
    send_file, url_for,
)
from flask_login import current_user

import data_store
from admin import is_api_request, roles_required

sgep_bp = Blueprint("sgep", __name__, url_prefix="/sgep")

LECTURA = ("admin", "pm", "auditor")
GESTION = ("admin", "pm")

TIPOS_ENCUESTA = {
    "satisfaccion": "Satisfacción",
    "avance": "Avance",
    "lecciones": "Lecciones aprendidas",
    "cierre": "Cierre",
}
TIPOS_PREGUNTA = {
    "nps": "NPS (0–10)",
    "avance": "% de avance (0–100)",
    "escala": "Escala 1–5",
    "si_no": "Sí / No",
    "opcion": "Opción múltiple",
    "texto": "Texto libre",
}
RANGOS = {"nps": (0, 10), "avance": (0, 100), "escala": (1, 5)}
EMAIL_RE = re.compile(r"^[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+$")
FECHA_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Endpoints de RAC (fuera del blueprint admin) que pm y auditor no deben usar
RAC_SOLO = {"detector", "predict", "reset_buffer", "api_alarm"}


# ═══════════════════════════════════════
#  DATOS (JSON en data/ vía data_store)
# ═══════════════════════════════════════

def cargar(nombre):
    return data_store._load_json(f"sgep_{nombre}.json")


def guardar(nombre, datos):
    data_store._save_json(f"sgep_{nombre}.json", datos)


def nuevo_id():
    return uuid.uuid4().hex[:12]


def buscar(lista, id_):
    return next((x for x in lista if x["id"] == id_), None)


def ahora():
    return datetime.now().isoformat(timespec="seconds")


# ═══════════════════════════════════════
#  KPIs (funciones puras, testeadas)
# ═══════════════════════════════════════

def nps(puntajes):
    """Net Promoter Score: % promotores (9–10) − % detractores (0–6). None sin datos."""
    if not puntajes:
        return None
    promotores = sum(1 for p in puntajes if p >= 9)
    detractores = sum(1 for p in puntajes if p <= 6)
    return round((promotores - detractores) * 100 / len(puntajes))


def promedio(valores):
    return sum(valores) / len(valores) if valores else None


def avance_esperado(proyecto, hoy):
    """% del plazo transcurrido entre inicio y fin del proyecto (0–100)."""
    ini, fin = date.fromisoformat(proyecto["inicio"]), date.fromisoformat(proyecto["fin"])
    if fin <= ini:
        return 100.0
    return max(0.0, min(100.0, (hoy - ini).days * 100 / (fin - ini).days))


def cumplimiento(avance_percibido, esperado):
    """Avance percibido sobre el esperado por calendario, tope 100 %."""
    if avance_percibido is None:
        return None
    if esperado <= 0:
        return 100.0
    return min(100.0, avance_percibido * 100 / esperado)


def participacion(invitaciones):
    if not invitaciones:
        return None
    return round(sum(1 for i in invitaciones if i.get("respondida")) * 100 / len(invitaciones))


def semaforo(nps_v, cumpl):
    """rojo: NPS negativo o cumplimiento < 60 %; amarillo: NPS < 30 o cumplimiento < 85 %."""
    if nps_v is None and cumpl is None:
        return "gris"
    if (nps_v is not None and nps_v < 0) or (cumpl is not None and cumpl < 60):
        return "rojo"
    if (nps_v is not None and nps_v < 30) or (cumpl is not None and cumpl < 85):
        return "amarillo"
    return "verde"


def _redondo(v):
    return None if v is None else round(v)


def _semana(fecha_iso):
    d = date.fromisoformat(fecha_iso[:10])
    return (d - timedelta(days=d.weekday())).isoformat()


def calcular_dashboard(filtros, hoy=None):
    """KPIs, filas por proyecto y series de gráficos según los filtros."""
    hoy = hoy or date.today()
    f = {k: (filtros.get(k) or "") for k in ("proyecto", "cliente", "responsable", "tipo", "desde", "hasta")}
    responsables = {r["id"]: r for r in cargar("responsables")}
    proyectos = [
        p for p in cargar("proyectos")
        if (not f["proyecto"] or p["id"] == f["proyecto"])
        and (not f["cliente"] or p.get("cliente") == f["cliente"])
        and (not f["responsable"] or p.get("responsable_id") == f["responsable"])
    ]
    pids = {p["id"] for p in proyectos}
    encuestas = {
        e["id"]: e for e in cargar("encuestas")
        if e["proyecto_id"] in pids and (not f["tipo"] or e["tipo"] == f["tipo"])
    }

    def en_rango(fecha_iso):
        d = fecha_iso[:10]
        return (not f["desde"] or d >= f["desde"]) and (not f["hasta"] or d <= f["hasta"])

    respuestas = [r for r in cargar("respuestas") if r["encuesta_id"] in encuestas and en_rango(r["fecha"])]
    invitaciones = [i for i in cargar("invitaciones") if i["encuesta_id"] in encuestas and en_rango(i["creada"])]

    def valores(rs, tipo):
        out = []
        for r in rs:
            for p in encuestas[r["encuesta_id"]]["preguntas"]:
                if p["tipo"] == tipo and p["id"] in r["valores"]:
                    out.append(r["valores"][p["id"]])
        return out

    por_proyecto = defaultdict(list)
    for r in respuestas:
        por_proyecto[encuestas[r["encuesta_id"]]["proyecto_id"]].append(r)
    inv_por_proyecto = defaultdict(list)
    for i in invitaciones:
        inv_por_proyecto[encuestas[i["encuesta_id"]]["proyecto_id"]].append(i)

    proyecto_de = {p["id"]: p for p in proyectos}

    def avances(rs):
        """(fecha, % percibido, % esperado a esa fecha) de cada respuesta de avance."""
        out = []
        for r in rs:
            e = encuestas[r["encuesta_id"]]
            esp = avance_esperado(proyecto_de[e["proyecto_id"]], date.fromisoformat(r["fecha"][:10]))
            out += [(r["fecha"], r["valores"][q["id"]], esp) for q in e["preguntas"]
                    if q["tipo"] == "avance" and q["id"] in r["valores"]]
        return sorted(out)

    filas = []
    for p in proyectos:
        rs = por_proyecto[p["id"]]
        n = nps(valores(rs, "nps"))
        av_rs = avances(rs)
        av = av_rs[-1][1] if av_rs else None  # la percepción más reciente
        esp = avance_esperado(p, hoy)
        # cada respuesta se compara con lo esperado el día que se respondió
        cu = promedio([cumplimiento(v, e) for _, v, e in av_rs])
        filas.append({
            "id": p["id"], "nombre": p["nombre"], "curso": p.get("curso", ""),
            "cliente": p.get("cliente", ""),
            "responsable": responsables.get(p.get("responsable_id"), {}).get("nombre", "—"),
            "respuestas": len(rs), "nps": n, "avance": _redondo(av), "esperado": round(esp),
            "cumplimiento": _redondo(cu), "participacion": participacion(inv_por_proyecto[p["id"]]),
            "semaforo": semaforo(n, cu),
        })

    puntajes_nps = valores(respuestas, "nps")
    semanas = sorted({_semana(r["fecha"]) for r in respuestas})
    por_semana = defaultdict(list)
    for r in respuestas:
        por_semana[_semana(r["fecha"])].append(r)

    calor = []
    for p in proyectos:
        celdas = []
        for s in semanas:
            rs = [r for r in por_semana[s] if encuestas[r["encuesta_id"]]["proyecto_id"] == p["id"]]
            celdas.append(_redondo(promedio(valores(rs, "avance"))))
        calor.append({"proyecto": p["nombre"], "valores": celdas})

    comentarios = []
    for r in sorted(respuestas, key=lambda r: r["fecha"], reverse=True):
        e = encuestas[r["encuesta_id"]]
        for p in e["preguntas"]:
            if p["tipo"] == "texto" and r["valores"].get(p["id"]):
                comentarios.append({
                    "fecha": r["fecha"][:10], "proyecto": next(x["nombre"] for x in proyectos if x["id"] == e["proyecto_id"]),
                    "pregunta": p["texto"], "texto": r["valores"][p["id"]],
                })
        if len(comentarios) >= 8:
            break

    return {
        "kpis": {
            "nps": nps(puntajes_nps),
            "avance": _redondo(promedio([x["avance"] for x in filas if x["avance"] is not None])),
            "cumplimiento": _redondo(promedio([x["cumplimiento"] for x in filas if x["cumplimiento"] is not None])),
            "participacion": participacion(invitaciones),
            "respuestas": len(respuestas),
            "en_riesgo": sum(1 for x in filas if x["semaforo"] == "rojo"),
        },
        "proyectos": filas,
        "torta": {
            "promotores": sum(1 for v in puntajes_nps if v >= 9),
            "pasivos": sum(1 for v in puntajes_nps if 7 <= v <= 8),
            "detractores": sum(1 for v in puntajes_nps if v <= 6),
        },
        "tendencia": {
            "semanas": semanas,
            "nps": [nps(valores(por_semana[s], "nps")) for s in semanas],
            "avance": [_redondo(promedio(valores(por_semana[s], "avance"))) for s in semanas],
        },
        "calor": {"semanas": semanas, "filas": calor},
        "por_tipo": dict(Counter(encuestas[r["encuesta_id"]]["tipo"] for r in respuestas)),
        "comentarios": comentarios[:8],
        "actualizado": ahora(),
    }


# ═══════════════════════════════════════
#  ENCUESTAS: validación de preguntas y respuestas
# ═══════════════════════════════════════

def parsear_preguntas(texto_json):
    """Valida las preguntas que manda el editor. Devuelve (preguntas, error)."""
    try:
        crudas = json.loads(texto_json or "[]")
    except ValueError:
        return None, "Formato de preguntas inválido."
    if not isinstance(crudas, list) or not crudas:
        return None, "La encuesta necesita al menos una pregunta."
    if len(crudas) > 50:
        return None, "Máximo 50 preguntas por encuesta."
    preguntas, ids = [], set()
    for c in crudas:
        if not isinstance(c, dict):
            return None, "Formato de preguntas inválido."
        texto = str(c.get("texto", "")).strip()[:300]
        tipo = c.get("tipo")
        if not texto or tipo not in TIPOS_PREGUNTA:
            return None, "Cada pregunta necesita texto y un tipo válido."
        pid = c.get("id")
        if not (isinstance(pid, str) and re.fullmatch(r"[a-z0-9]{1,16}", pid)) or pid in ids:
            pid = nuevo_id()
        opciones = []
        if tipo == "opcion":
            opciones = [str(o).strip()[:100] for o in (c.get("opciones") or []) if str(o).strip()][:20]
            if len(opciones) < 2:
                return None, f"«{texto}» necesita al menos dos opciones."
        condicion = c.get("condicion")
        if condicion:
            # Solo puede depender de una pregunta anterior: así no hay ciclos
            if not isinstance(condicion, dict) or condicion.get("pregunta") not in ids:
                return None, f"La condición de «{texto}» debe apuntar a una pregunta anterior."
            condicion = {"pregunta": condicion["pregunta"], "valor": str(condicion.get("valor", "")).strip()[:100]}
        preguntas.append({"id": pid, "texto": texto, "tipo": tipo, "opciones": opciones,
                          "condicion": condicion or None})
        ids.add(pid)
    return preguntas, None


def visible(pregunta, valores):
    c = pregunta.get("condicion")
    return not c or str(valores.get(c["pregunta"], "")) == c["valor"]


def validar_respuesta(encuesta, form):
    """Valida lo que manda quien responde. Solo guarda preguntas visibles. Devuelve (valores, error)."""
    valores = {}
    for p in encuesta["preguntas"]:  # en orden: las condiciones apuntan a preguntas anteriores
        if not visible(p, valores):
            continue
        crudo = (form.get(p["id"]) or "").strip()
        if not crudo:
            if p["tipo"] == "texto":
                continue
            return None, f"Falta responder: {p['texto']}"
        if p["tipo"] in RANGOS:
            lo, hi = RANGOS[p["tipo"]]
            try:
                v = int(crudo)
            except ValueError:
                return None, f"Valor inválido en: {p['texto']}"
            if not lo <= v <= hi:
                return None, f"Valor fuera de rango en: {p['texto']}"
        elif p["tipo"] == "si_no":
            if crudo not in ("si", "no"):
                return None, f"Valor inválido en: {p['texto']}"
            v = crudo
        elif p["tipo"] == "opcion":
            if crudo not in p["opciones"]:
                return None, f"Opción inválida en: {p['texto']}"
            v = crudo
        else:
            v = crudo[:1000]
        valores[p["id"]] = v
    return valores, None


# ═══════════════════════════════════════
#  CORREO (SMTP opcional; siempre queda registro)
# ═══════════════════════════════════════

def enviar_correo(para, asunto, cuerpo):
    """Envía por SMTP si SMTP_HOST está configurado; si no, solo lo registra (modo demo)."""
    registro = {"id": nuevo_id(), "fecha": ahora(), "para": para, "asunto": asunto,
                "cuerpo": cuerpo, "enviado": False, "error": ""}
    host = os.environ.get("SMTP_HOST", "")
    if host:
        try:
            msg = EmailMessage()
            msg["From"] = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER", "")
            msg["To"] = para
            msg["Subject"] = asunto
            msg.set_content(cuerpo)
            with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=15) as s:
                s.starttls()
                if os.environ.get("SMTP_USER"):
                    s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
                s.send_message(msg)
            registro["enviado"] = True
        except (OSError, ValueError, smtplib.SMTPException) as e:
            registro["error"] = str(e)[:200]
    correos = cargar("correos")
    correos.insert(0, registro)
    guardar("correos", correos[:500])
    return registro


def _correo_invitacion(encuesta, proyecto, inv, recordatorio=False):
    link = url_for("sgep.responder", token=inv["token"], _external=True)
    asunto = ("Recordatorio: " if recordatorio else "") + f"Encuesta «{encuesta['titulo']}» — {proyecto['nombre']}"
    cuerpo = (
        f"Hola {inv.get('nombre') or ''},\n\n"
        f"Te invitamos a responder la encuesta «{encuesta['titulo']}» del proyecto {proyecto['nombre']}.\n"
        f"Tu link personal (no lo compartas): {link}\n\n"
        + ("La encuesta es anónima: solo registramos que respondiste, no qué respondiste.\n\n" if encuesta.get("anonima") else "")
        + "Gracias.\nSGEP — Expotécnica"
    )
    return enviar_correo(inv["email"], asunto, cuerpo)


# ═══════════════════════════════════════
#  ROLES: pm y auditor solo usan el SGEP
# ═══════════════════════════════════════

@sgep_bp.before_app_request
def confinar_roles_sgep():
    if not current_user.is_authenticated or current_user.role not in ("pm", "auditor"):
        return None
    ep = request.endpoint or ""
    if ep in ("admin.login", "admin.logout") or not (ep.startswith("admin.") or ep in RAC_SOLO):
        return None
    if is_api_request():
        return jsonify({"error": "Permisos insuficientes"}), 403
    return redirect(url_for("sgep.dashboard"))


@sgep_bp.app_context_processor
def tipos_sgep():
    return {"TIPOS_ENCUESTA": TIPOS_ENCUESTA, "TIPOS_PREGUNTA": TIPOS_PREGUNTA,
            "puede_gestionar": getattr(current_user, "role", None) in GESTION}


# ═══════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════

def _opciones_filtros():
    proyectos = cargar("proyectos")
    return {
        "proyectos": proyectos,
        "clientes": sorted({p.get("cliente", "") for p in proyectos if p.get("cliente")}),
        "responsables": cargar("responsables"),
    }


@sgep_bp.route("/")
@roles_required(*LECTURA)
def dashboard():
    return render_template("sgep/dashboard.html", **_opciones_filtros())


@sgep_bp.route("/api/dashboard")
@roles_required(*LECTURA)
def api_dashboard():
    return jsonify(calcular_dashboard(request.args))


# ═══════════════════════════════════════
#  MÓDULO 1: PROYECTOS Y RESPONSABLES
# ═══════════════════════════════════════

@sgep_bp.route("/proyectos")
@roles_required(*LECTURA)
def proyectos():
    responsables = cargar("responsables")
    nombres = {r["id"]: r["nombre"] for r in responsables}
    encuestas = cargar("encuestas")
    lista = [dict(p, responsable=nombres.get(p.get("responsable_id"), "—"),
                  encuestas=sum(1 for e in encuestas if e["proyecto_id"] == p["id"]))
             for p in cargar("proyectos")]
    return render_template("sgep/proyectos.html", proyectos=lista, responsables=responsables)


@sgep_bp.route("/proyectos/guardar", methods=["POST"])
@roles_required(*GESTION)
def guardar_proyecto():
    form = request.form
    datos = {k: form.get(k, "").strip()[:200] for k in ("nombre", "curso", "cliente", "responsable_id", "inicio", "fin", "descripcion")}
    if not datos["nombre"] or not datos["curso"]:
        flash("Nombre y curso son obligatorios.", "error")
    elif not (FECHA_RE.match(datos["inicio"]) and FECHA_RE.match(datos["fin"])) or datos["fin"] < datos["inicio"]:
        flash("Fechas inválidas: el fin no puede ser anterior al inicio.", "error")
    elif not buscar(cargar("responsables"), datos["responsable_id"]):
        flash("Elegí un responsable.", "error")
    else:
        lista = cargar("proyectos")
        existente = buscar(lista, form.get("id", ""))
        if existente:
            existente.update(datos)
        else:
            lista.append(dict(datos, id=nuevo_id(), creado=ahora()))
        guardar("proyectos", lista)
        flash(f"Proyecto «{datos['nombre']}» guardado.", "success")
    return redirect(url_for("sgep.proyectos"))


@sgep_bp.route("/proyectos/<pid>/borrar", methods=["POST"])
@roles_required(*GESTION)
def borrar_proyecto(pid):
    if any(e["proyecto_id"] == pid for e in cargar("encuestas")):
        flash("El proyecto tiene encuestas: borralas primero.", "error")
    else:
        guardar("proyectos", [p for p in cargar("proyectos") if p["id"] != pid])
        flash("Proyecto eliminado.", "success")
    return redirect(url_for("sgep.proyectos"))


@sgep_bp.route("/responsables/guardar", methods=["POST"])
@roles_required(*GESTION)
def guardar_responsable():
    nombre = request.form.get("nombre", "").strip()[:120]
    email = request.form.get("email", "").strip()[:200]
    if not nombre or not EMAIL_RE.match(email):
        flash("Nombre y un email válido son obligatorios.", "error")
    else:
        lista = cargar("responsables")
        existente = buscar(lista, request.form.get("id", ""))
        if existente:
            existente.update(nombre=nombre, email=email)
        else:
            lista.append({"id": nuevo_id(), "nombre": nombre, "email": email})
        guardar("responsables", lista)
        flash(f"Responsable «{nombre}» guardado.", "success")
    return redirect(url_for("sgep.proyectos"))


@sgep_bp.route("/responsables/<rid>/borrar", methods=["POST"])
@roles_required(*GESTION)
def borrar_responsable(rid):
    if any(p.get("responsable_id") == rid for p in cargar("proyectos")):
        flash("El responsable tiene proyectos asignados.", "error")
    else:
        guardar("responsables", [r for r in cargar("responsables") if r["id"] != rid])
        flash("Responsable eliminado.", "success")
    return redirect(url_for("sgep.proyectos"))


# ═══════════════════════════════════════
#  MÓDULO 2: CONSTRUCTOR DE ENCUESTAS Y PLANTILLAS
# ═══════════════════════════════════════

@sgep_bp.route("/encuestas")
@roles_required(*LECTURA)
def encuestas():
    proyectos = {p["id"]: p for p in cargar("proyectos")}
    respuestas = Counter(r["encuesta_id"] for r in cargar("respuestas"))
    invitaciones = defaultdict(list)
    for i in cargar("invitaciones"):
        invitaciones[i["encuesta_id"]].append(i)
    lista = [dict(e, proyecto=proyectos.get(e["proyecto_id"], {}).get("nombre", "—"),
                  n_respuestas=respuestas[e["id"]], participacion=participacion(invitaciones[e["id"]]))
             for e in cargar("encuestas")]
    return render_template("sgep/encuestas.html", encuestas=lista, plantillas=cargar("plantillas"),
                           proyectos=list(proyectos.values()))


@sgep_bp.route("/encuestas/nueva", methods=["POST"])
@roles_required(*GESTION)
def nueva_encuesta():
    proyecto = buscar(cargar("proyectos"), request.form.get("proyecto_id", ""))
    if not proyecto:
        flash("Elegí un proyecto.", "error")
        return redirect(url_for("sgep.encuestas"))
    plantilla = buscar(cargar("plantillas"), request.form.get("plantilla_id", ""))
    base = plantilla or {"titulo": "Nueva encuesta", "tipo": "satisfaccion", "anonima": False,
                         "preguntas": [{"id": nuevo_id(), "texto": "¿Qué tan probable es que recomiendes este proyecto?",
                                        "tipo": "nps", "opciones": [], "condicion": None}]}
    encuesta = {
        "id": nuevo_id(), "proyecto_id": proyecto["id"],
        "titulo": request.form.get("titulo", "").strip()[:150] or base["titulo"],
        "tipo": base["tipo"], "anonima": base["anonima"],
        "preguntas": json.loads(json.dumps(base["preguntas"])),  # copia profunda
        "activa": True, "link": secrets.token_urlsafe(12), "creada": ahora(),
    }
    lista = cargar("encuestas")
    lista.append(encuesta)
    guardar("encuestas", lista)
    return redirect(url_for("sgep.editar_encuesta", eid=encuesta["id"]))


@sgep_bp.route("/encuestas/<eid>/editar", methods=["GET", "POST"])
@roles_required(*GESTION)
def editar_encuesta(eid):
    lista = cargar("encuestas")
    encuesta = buscar(lista, eid) or abort(404)
    if request.method == "POST":
        preguntas, error = parsear_preguntas(request.form.get("preguntas_json"))
        tipo = request.form.get("tipo")
        proyecto = buscar(cargar("proyectos"), request.form.get("proyecto_id", ""))
        if error or tipo not in TIPOS_ENCUESTA or not proyecto:
            flash(error or "Tipo o proyecto inválido.", "error")
        else:
            encuesta.update(
                titulo=request.form.get("titulo", "").strip()[:150] or encuesta["titulo"],
                tipo=tipo, proyecto_id=proyecto["id"], preguntas=preguntas,
                anonima=request.form.get("anonima") == "on",
            )
            guardar("encuestas", lista)
            flash("Encuesta guardada.", "success")
            if request.form.get("como_plantilla") == "on":
                plantillas = cargar("plantillas")
                plantillas.append({"id": nuevo_id(), "titulo": encuesta["titulo"], "tipo": tipo,
                                   "anonima": encuesta["anonima"], "preguntas": preguntas})
                guardar("plantillas", plantillas)
                flash("También quedó guardada como plantilla.", "success")
            return redirect(url_for("sgep.encuesta", eid=eid))
    tiene_respuestas = any(r["encuesta_id"] == eid for r in cargar("respuestas"))
    return render_template("sgep/editor.html", encuesta=encuesta, proyectos=cargar("proyectos"),
                           tiene_respuestas=tiene_respuestas)


@sgep_bp.route("/encuestas/<eid>/borrar", methods=["POST"])
@roles_required(*GESTION)
def borrar_encuesta(eid):
    guardar("encuestas", [e for e in cargar("encuestas") if e["id"] != eid])
    guardar("invitaciones", [i for i in cargar("invitaciones") if i["encuesta_id"] != eid])
    guardar("respuestas", [r for r in cargar("respuestas") if r["encuesta_id"] != eid])
    flash("Encuesta eliminada junto con sus invitaciones y respuestas.", "success")
    return redirect(url_for("sgep.encuestas"))


@sgep_bp.route("/plantillas/<tid>/borrar", methods=["POST"])
@roles_required(*GESTION)
def borrar_plantilla(tid):
    guardar("plantillas", [t for t in cargar("plantillas") if t["id"] != tid])
    flash("Plantilla eliminada.", "success")
    return redirect(url_for("sgep.encuestas"))


# ═══════════════════════════════════════
#  MÓDULO 3: DISTRIBUCIÓN Y RECOLECCIÓN
# ═══════════════════════════════════════

@sgep_bp.route("/encuestas/<eid>")
@roles_required(*LECTURA)
def encuesta(eid):
    e = buscar(cargar("encuestas"), eid) or abort(404)
    proyecto = buscar(cargar("proyectos"), e["proyecto_id"]) or {"nombre": "—"}
    invitaciones = [i for i in cargar("invitaciones") if i["encuesta_id"] == eid]
    n_respuestas = sum(1 for r in cargar("respuestas") if r["encuesta_id"] == eid)
    return render_template(
        "sgep/encuesta.html", encuesta=e, proyecto=proyecto, invitaciones=invitaciones,
        n_respuestas=n_respuestas, participacion=participacion(invitaciones),
        link=url_for("sgep.responder", token=e["link"], _external=True),
    )


@sgep_bp.route("/encuestas/<eid>/invitar", methods=["POST"])
@roles_required(*GESTION)
def invitar(eid):
    e = buscar(cargar("encuestas"), eid) or abort(404)
    proyecto = buscar(cargar("proyectos"), e["proyecto_id"]) or abort(404)
    lista = cargar("invitaciones")
    ya = {i["email"].lower() for i in lista if i["encuesta_id"] == eid}
    nuevas, invalidas = 0, []
    # Una línea por persona: "Nombre <email>" o solo "email"
    for linea in request.form.get("destinatarios", "").splitlines()[:200]:
        linea = linea.strip()
        if not linea:
            continue
        m = re.match(r"^(.*?)<([^>]+)>$", linea)
        nombre, email = (m.group(1).strip(), m.group(2).strip()) if m else ("", linea)
        if not EMAIL_RE.match(email):
            invalidas.append(linea[:60])
            continue
        if email.lower() in ya:
            continue
        inv = {"id": nuevo_id(), "encuesta_id": eid, "nombre": nombre[:120], "email": email,
               "token": secrets.token_urlsafe(16), "creada": ahora(), "recordatorios": 0,
               "respondida": None}
        _correo_invitacion(e, proyecto, inv)
        lista.append(inv)
        ya.add(email.lower())
        nuevas += 1
    guardar("invitaciones", lista)
    flash(f"{nuevas} invitación(es) enviada(s).", "success")
    if invalidas:
        flash("Emails inválidos (no se enviaron): " + ", ".join(invalidas[:5]), "warning")
    return redirect(url_for("sgep.encuesta", eid=eid))


@sgep_bp.route("/encuestas/<eid>/recordar", methods=["POST"])
@roles_required(*GESTION)
def recordar(eid):
    e = buscar(cargar("encuestas"), eid) or abort(404)
    proyecto = buscar(cargar("proyectos"), e["proyecto_id"]) or abort(404)
    lista = cargar("invitaciones")
    pendientes = [i for i in lista if i["encuesta_id"] == eid and not i["respondida"]]
    for inv in pendientes:
        _correo_invitacion(e, proyecto, inv, recordatorio=True)
        inv["recordatorios"] = inv.get("recordatorios", 0) + 1
    guardar("invitaciones", lista)
    flash(f"Recordatorio enviado a {len(pendientes)} persona(s) sin responder.", "success")
    return redirect(url_for("sgep.encuesta", eid=eid))


@sgep_bp.route("/encuestas/<eid>/estado", methods=["POST"])
@roles_required(*GESTION)
def cambiar_estado(eid):
    lista = cargar("encuestas")
    e = buscar(lista, eid) or abort(404)
    e["activa"] = not e.get("activa", True)
    guardar("encuestas", lista)
    flash("Encuesta " + ("abierta." if e["activa"] else "cerrada: ya no acepta respuestas."), "success")
    return redirect(url_for("sgep.encuesta", eid=eid))


@sgep_bp.route("/correos")
@roles_required(*LECTURA)
def correos():
    return render_template("sgep/correos.html", correos=cargar("correos")[:200],
                           smtp=bool(os.environ.get("SMTP_HOST")))


@sgep_bp.route("/r/<token>", methods=["GET", "POST"])
def responder(token):
    """Página pública: link único por invitación o link general de la encuesta (QR)."""
    invitaciones = cargar("invitaciones")
    inv = next((i for i in invitaciones if i["token"] == token), None)
    encuestas_ = cargar("encuestas")
    e = buscar(encuestas_, inv["encuesta_id"]) if inv else next((x for x in encuestas_ if x["link"] == token), None)
    if not e:
        abort(404)
    proyecto = buscar(cargar("proyectos"), e["proyecto_id"]) or {"nombre": "—"}
    estado, error = None, None
    if not e.get("activa", True):
        estado = "cerrada"
    elif inv and inv["respondida"]:
        estado = "ya"
    elif request.method == "POST":
        valores, error = validar_respuesta(e, request.form)
        if not error:
            # shortcut: el link general (QR) admite varias respuestas por persona; si hace falta una sola, usar invitaciones
            respondente = None if e.get("anonima") or not inv else (inv.get("nombre") or inv["email"])
            respuestas = cargar("respuestas")
            respuestas.append({"id": nuevo_id(), "encuesta_id": e["id"], "proyecto_id": e["proyecto_id"],
                               "fecha": ahora(), "respondente": respondente, "valores": valores})
            guardar("respuestas", respuestas)
            if inv:
                inv["respondida"] = ahora()
                guardar("invitaciones", invitaciones)
            estado = "gracias"
    return render_template("sgep/responder.html", encuesta=e, proyecto=proyecto, estado=estado,
                           error=error, previo=request.form), (400 if error else 200)


# ═══════════════════════════════════════
#  MÓDULO 5: REPORTES (Excel y PDF)
# ═══════════════════════════════════════

COLUMNAS = [("nombre", "Proyecto"), ("curso", "Curso"), ("cliente", "Cliente"), ("responsable", "Responsable"),
            ("respuestas", "Respuestas"), ("nps", "NPS"), ("avance", "Avance %"), ("esperado", "Esperado %"),
            ("cumplimiento", "Cumplimiento %"), ("participacion", "Participación %"), ("semaforo", "Semáforo")]


def _texto(v):
    return "—" if v is None else str(v)


@sgep_bp.route("/export/excel")
@roles_required(*LECTURA)
def exportar_excel():
    from openpyxl import Workbook
    from openpyxl.styles import Font

    datos = calcular_dashboard(request.args)
    wb = Workbook()
    hoja = wb.active
    hoja.title = "KPIs"
    hoja.append(["Indicador", "Valor"])
    for k, v in datos["kpis"].items():
        hoja.append([k, v])
    hoja = wb.create_sheet("Proyectos")
    hoja.append([t for _, t in COLUMNAS])
    for fila in datos["proyectos"]:
        hoja.append([fila[k] for k, _ in COLUMNAS])
    # Formato largo (una fila por respuesta a pregunta): listo para tablas dinámicas
    hoja = wb.create_sheet("Respuestas")
    hoja.append(["Fecha", "Proyecto", "Encuesta", "Tipo", "Respondente", "Pregunta", "Valor"])
    ids = {f["id"]: f["nombre"] for f in datos["proyectos"]}
    encuestas_ = {e["id"]: e for e in cargar("encuestas") if e["proyecto_id"] in ids}
    for r in cargar("respuestas"):
        e = encuestas_.get(r["encuesta_id"])
        if not e or (request.args.get("tipo") and e["tipo"] != request.args["tipo"]):
            continue
        d = r["fecha"][:10]
        if (request.args.get("desde") and d < request.args["desde"]) or (request.args.get("hasta") and d > request.args["hasta"]):
            continue
        for p in e["preguntas"]:
            if p["id"] in r["valores"]:
                hoja.append([r["fecha"], ids[e["proyecto_id"]], e["titulo"], TIPOS_ENCUESTA[e["tipo"]],
                             r["respondente"] or "Anónimo", p["texto"], r["valores"][p["id"]]])
    for ws in wb.worksheets:
        for celda in ws[1]:
            celda.font = Font(bold=True)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"sgep_{date.today()}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@sgep_bp.route("/export/pdf")
@roles_required(*LECTURA)
def exportar_pdf():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    datos = calcular_dashboard(request.args)
    estilos = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), title="Reporte SGEP")
    k = datos["kpis"]
    filtros = ", ".join(f"{c}={v}" for c, v in request.args.items() if v) or "sin filtros"
    elementos = [
        Paragraph("SGEP — Reporte de evaluación de proyectos", estilos["Title"]),
        Paragraph(f"Generado {datos['actualizado']} · {filtros}", estilos["Normal"]),
        Spacer(1, 12),
        Table([["NPS", "Avance percibido", "Cumplimiento", "Participación", "Respuestas", "En riesgo"],
               [_texto(k["nps"]), f"{_texto(k['avance'])} %", f"{_texto(k['cumplimiento'])} %",
                f"{_texto(k['participacion'])} %", k["respuestas"], k["en_riesgo"]]],
              style=[("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                     ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8751A")),
                     ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]),
        Spacer(1, 16),
    ]
    color_semaforo = {"rojo": "#FCA5A5", "amarillo": "#FDE68A", "verde": "#A7F3D0", "gris": "#E5E7EB"}
    tabla = [[t for _, t in COLUMNAS]] + [[_texto(f[c]) for c, _ in COLUMNAS] for f in datos["proyectos"]]
    estilo = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                         ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                         ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                         ("FONTSIZE", (0, 0), (-1, -1), 8)])
    for n, f in enumerate(datos["proyectos"], start=1):
        estilo.add("BACKGROUND", (-1, n), (-1, n), colors.HexColor(color_semaforo[f["semaforo"]]))
    elementos.append(Table(tabla, style=estilo, repeatRows=1))
    doc.build(elementos)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"sgep_{date.today()}.pdf",
                     mimetype="application/pdf")
