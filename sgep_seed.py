"""Datos de prueba del SGEP-Dashboard (solo para desarrollo local y la demo).

Uso:  python sgep_seed.py [--force]

Crea responsables, proyectos de 7mo, plantillas, encuestas, invitaciones y
~10 semanas de respuestas en data/sgep_*.json, y dos usuarios demo:
    pm_demo / pm-demo-2026            (rol pm)
    auditor_demo / auditor-demo-2026  (rol auditor)
No correr en producción: --force pisa los datos del SGEP que haya.
"""
import random
import secrets
import sys
from datetime import date, datetime, time, timedelta

import data_store
import sgep

USUARIOS_DEMO = [("pm_demo", "pm-demo-2026", "pm"), ("auditor_demo", "auditor-demo-2026", "auditor")]

PLANTILLAS = [
    {"id": "plsatisf", "titulo": "Satisfacción del cliente", "tipo": "satisfaccion", "anonima": True, "preguntas": [
        {"id": "nps1", "texto": "¿Qué tan probable es que recomiendes este proyecto?", "tipo": "nps", "opciones": [], "condicion": None},
        {"id": "com1", "texto": "La comunicación con el equipo fue buena", "tipo": "escala", "opciones": [], "condicion": None},
        {"id": "ret1", "texto": "¿Hubo retrasos en las entregas?", "tipo": "si_no", "opciones": [], "condicion": None},
        {"id": "cau1", "texto": "¿Cuál fue la causa del retraso?", "tipo": "texto", "opciones": [],
         "condicion": {"pregunta": "ret1", "valor": "si"}},
    ]},
    {"id": "plavance", "titulo": "Avance percibido", "tipo": "avance", "anonima": False, "preguntas": [
        {"id": "ava1", "texto": "¿Qué porcentaje del proyecto creés que está terminado?", "tipo": "avance", "opciones": [], "condicion": None},
        {"id": "rie1", "texto": "¿Cuál es hoy el mayor riesgo?", "tipo": "opcion",
         "opciones": ["Tiempo", "Presupuesto", "Materiales", "Equipo"], "condicion": None},
        {"id": "con1", "texto": "Confío en que llegamos a la Expotécnica", "tipo": "escala", "opciones": [], "condicion": None},
    ]},
    {"id": "pllecc", "titulo": "Lecciones aprendidas", "tipo": "lecciones", "anonima": True, "preguntas": [
        {"id": "bien1", "texto": "¿Qué salió bien?", "tipo": "texto", "opciones": [], "condicion": None},
        {"id": "dist1", "texto": "¿Qué harías distinto la próxima vez?", "tipo": "texto", "opciones": [], "condicion": None},
        {"id": "nps2", "texto": "¿Recomendarías trabajar con este equipo?", "tipo": "nps", "opciones": [], "condicion": None},
    ]},
]

RESPONSABLES = [
    ("Prof. Laura Gómez", "laura.gomez@example.com"),
    ("Prof. Martín Ruiz", "martin.ruiz@example.com"),
    ("Ing. Carla Sosa", "carla.sosa@example.com"),
    ("Prof. Diego Fernández", "diego.fernandez@example.com"),
]

# (nombre, curso, cliente, responsable, calidad 0–1: qué tan bien le va)
PROYECTOS = [
    ("RAC Vision — casco inteligente", "7mo 1ra", "Cooperativa minera", 0, 0.9),
    ("Invernadero automatizado", "7mo 1ra", "Municipalidad", 1, 0.75),
    ("Brazo robótico clasificador", "7mo 2da", "Empresa local", 2, 0.45),
    ("Estación meteorológica IoT", "7mo 2da", "Escuela", 3, 0.8),
    ("App de turnos para el SUM", "7mo 3ra", "Escuela", 1, 0.25),
    ("Huerta hidropónica", "7mo 3ra", "Municipalidad", 0, 0.6),
]

LECCIONES_BIEN = ["La división de tareas funcionó", "Buen trabajo con el cliente", "Prototipo temprano",
                  "Documentamos todo desde el principio", "Pruebas en el taller cada semana"]
LECCIONES_DISTINTO = ["Pedir materiales antes", "Probar antes de integrar", "Más reuniones con el cliente",
                      "Repartir mejor la carga", "Planificar el Gantt con margen"]
CAUSAS = ["Faltaron materiales", "Se rompió un componente", "Cambios de alcance del cliente", "Exámenes"]


def sembrar(hoy=None, semilla=7):
    rnd = random.Random(semilla)
    hoy = hoy or date.today()
    responsables = [{"id": f"resp{i}", "nombre": n, "email": e} for i, (n, e) in enumerate(RESPONSABLES)]
    proyectos, encuestas, invitaciones, respuestas = [], [], [], []
    for i, (nombre, curso, cliente, r, calidad) in enumerate(PROYECTOS):
        inicio = hoy - timedelta(days=75)
        fin = hoy + timedelta(days=20 + 10 * i)
        p = {"id": f"proy{i}", "nombre": nombre, "curso": curso, "cliente": cliente,
             "responsable_id": responsables[r]["id"], "inicio": inicio.isoformat(), "fin": fin.isoformat(),
             "descripcion": f"Proyecto de {curso} para {cliente}", "creado": f"{inicio}T08:00:00"}
        proyectos.append(p)
        for pl in PLANTILLAS:
            e = {"id": f"{p['id']}-{pl['id']}", "proyecto_id": p["id"], "titulo": pl["titulo"], "tipo": pl["tipo"],
                 "anonima": pl["anonima"], "preguntas": pl["preguntas"], "activa": True,
                 "link": secrets.token_urlsafe(12), "creada": f"{inicio}T09:00:00"}
            encuestas.append(e)
            for k in range(8):
                creada = inicio + timedelta(days=rnd.randint(0, 65))
                respondio = rnd.random() < 0.45 + 0.45 * calidad
                fecha = datetime.combine(min(hoy, creada + timedelta(days=rnd.randint(0, 9))),
                                         time(rnd.randint(8, 20), rnd.randint(0, 59)))
                inv = {"id": f"{e['id']}-i{k}", "encuesta_id": e["id"], "nombre": f"Persona {k + 1}",
                       "email": f"persona{k + 1}.{p['id']}@example.com", "token": secrets.token_urlsafe(16),
                       "creada": f"{creada}T10:00:00", "recordatorios": 0 if respondio else rnd.randint(0, 2),
                       "respondida": fecha.isoformat(timespec="seconds") if respondio else None}
                invitaciones.append(inv)
                if respondio:
                    respuestas.append({"id": f"{inv['id']}-r", "encuesta_id": e["id"], "proyecto_id": p["id"],
                                       "fecha": inv["respondida"],
                                       "respondente": None if e["anonima"] else inv["nombre"],
                                       "valores": _valores(rnd, pl["id"], calidad, p, fecha.date())})
    sgep.guardar("responsables", responsables)
    sgep.guardar("proyectos", proyectos)
    sgep.guardar("plantillas", PLANTILLAS)
    sgep.guardar("encuestas", encuestas)
    sgep.guardar("invitaciones", invitaciones)
    sgep.guardar("respuestas", respuestas)
    sgep.guardar("correos", [])
    for usuario, clave, rol in USUARIOS_DEMO:
        if not data_store.get_user_by_username(usuario):
            data_store.create_user(usuario, clave, rol)
    return {"proyectos": len(proyectos), "encuestas": len(encuestas), "respuestas": len(respuestas)}


def _valores(rnd, plantilla, calidad, proyecto, dia):
    def nps_v():
        return max(0, min(10, round(rnd.gauss(4 + 6 * calidad, 1.8))))
    if plantilla == "plsatisf":
        v = {"nps1": nps_v(), "com1": max(1, min(5, round(rnd.gauss(1.5 + 3.5 * calidad, 0.8)))),
             "ret1": "si" if rnd.random() > calidad else "no"}
        if v["ret1"] == "si":
            v["cau1"] = rnd.choice(CAUSAS)
        return v
    if plantilla == "plavance":
        esperado = sgep.avance_esperado(proyecto, dia)
        return {"ava1": max(0, min(100, round(esperado * (0.5 + 0.6 * calidad) + rnd.gauss(0, 6)))),
                "rie1": rnd.choice(["Tiempo", "Presupuesto", "Materiales", "Equipo"]),
                "con1": max(1, min(5, round(rnd.gauss(1.5 + 3.5 * calidad, 0.8))))}
    return {"bien1": rnd.choice(LECCIONES_BIEN), "dist1": rnd.choice(LECCIONES_DISTINTO), "nps2": nps_v()}


if __name__ == "__main__":
    if sgep.cargar("proyectos") and "--force" not in sys.argv:
        sys.exit("Ya hay datos del SGEP en data/. Usá --force para reemplazarlos por los de prueba.")
    print("Datos de prueba creados:", sembrar())
    print("Usuarios demo: pm_demo / pm-demo-2026 · auditor_demo / auditor-demo-2026")
