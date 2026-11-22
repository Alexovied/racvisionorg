"""Tests del SGEP-Dashboard: KPIs/NPS, permisos por rol, CSRF y flujo de respuesta."""
import json
from datetime import date

import pytest

import data_store
import sgep
import sgep_seed

HOY = date(2026, 11, 10)


# ── KPIs (funciones puras) ───────────────────────────────────────────────────

@pytest.mark.parametrize("puntajes,esperado", [
    ([], None),
    ([10, 10, 9], 100),
    ([0, 3, 6, 6], -100),
    ([10, 9, 8, 7, 6, 0], 0),          # 2 promotores, 2 pasivos, 2 detractores
    ([10, 10, 10, 8, 2], 40),          # 60 % − 20 %
    ([7, 8], 0),                        # solo pasivos
])
def test_nps(puntajes, esperado):
    assert sgep.nps(puntajes) == esperado


def test_avance_esperado_y_cumplimiento():
    p = {"inicio": "2026-01-01", "fin": "2026-01-11"}
    assert sgep.avance_esperado(p, date(2025, 12, 1)) == 0
    assert sgep.avance_esperado(p, date(2026, 1, 6)) == 50
    assert sgep.avance_esperado(p, date(2026, 3, 1)) == 100
    assert sgep.cumplimiento(40, 50) == 80
    assert sgep.cumplimiento(90, 50) == 100       # tope
    assert sgep.cumplimiento(None, 50) is None
    assert sgep.cumplimiento(10, 0) == 100


@pytest.mark.parametrize("nps_v,cumpl,color", [
    (None, None, "gris"), (50, 95, "verde"), (20, 95, "amarillo"), (50, 70, "amarillo"),
    (-5, 95, "rojo"), (50, 59, "rojo"), (None, 90, "verde"),
])
def test_semaforo(nps_v, cumpl, color):
    assert sgep.semaforo(nps_v, cumpl) == color


def test_participacion():
    assert sgep.participacion([]) is None
    assert sgep.participacion([{"respondida": "x"}, {"respondida": None}, {"respondida": None}, {"respondida": "y"}]) == 50


def test_dashboard_con_datos_de_prueba():
    sgep_seed.sembrar(hoy=HOY)
    d = sgep.calcular_dashboard({}, hoy=HOY)
    assert len(d["proyectos"]) == len(sgep_seed.PROYECTOS)
    assert -100 <= d["kpis"]["nps"] <= 100
    assert d["kpis"]["respuestas"] == len(data_store._load_json("sgep_respuestas.json"))
    t = d["torta"]
    assert t["promotores"] + t["pasivos"] + t["detractores"] > 0
    assert len(d["tendencia"]["semanas"]) == len(d["calor"]["semanas"]) >= 8
    colores = {p["semaforo"] for p in d["proyectos"]}
    assert "verde" in colores and "rojo" in colores   # la demo muestra el semáforo completo


def test_dashboard_filtros():
    sgep_seed.sembrar(hoy=HOY)
    solo = sgep.calcular_dashboard({"proyecto": "proy0"}, hoy=HOY)
    assert [p["id"] for p in solo["proyectos"]] == ["proy0"]
    avance = sgep.calcular_dashboard({"tipo": "avance"}, hoy=HOY)
    assert avance["kpis"]["nps"] is None and avance["kpis"]["avance"] is not None
    assert set(avance["por_tipo"]) == {"avance"}
    nada = sgep.calcular_dashboard({"desde": "2030-01-01"}, hoy=HOY)
    assert nada["kpis"]["respuestas"] == 0 and nada["kpis"]["nps"] is None
    cliente = sgep.calcular_dashboard({"cliente": "Escuela"}, hoy=HOY)
    assert {p["cliente"] for p in cliente["proyectos"]} == {"Escuela"}


# ── Preguntas y respuestas ───────────────────────────────────────────────────

ENCUESTA = {"preguntas": sgep_seed.PLANTILLAS[0]["preguntas"]}


def test_respuesta_valida_y_condicional():
    v, err = sgep.validar_respuesta(ENCUESTA, {"nps1": "9", "com1": "4", "ret1": "no", "cau1": "no debería guardarse"})
    assert err is None
    assert v == {"nps1": 9, "com1": 4, "ret1": "no"}           # cau1 oculta: se ignora
    v, err = sgep.validar_respuesta(ENCUESTA, {"nps1": "9", "com1": "4", "ret1": "si", "cau1": "materiales"})
    assert v["cau1"] == "materiales"


@pytest.mark.parametrize("form", [
    {"nps1": "11", "com1": "4", "ret1": "no"},
    {"nps1": "-1", "com1": "4", "ret1": "no"},
    {"nps1": "abc", "com1": "4", "ret1": "no"},
    {"nps1": "5", "com1": "6", "ret1": "no"},
    {"nps1": "5", "com1": "3", "ret1": "quizas"},
    {"com1": "3", "ret1": "no"},
])
def test_respuesta_invalida(form):
    v, err = sgep.validar_respuesta(ENCUESTA, form)
    assert v is None and err


def test_parsear_preguntas():
    ok, err = sgep.parsear_preguntas(json.dumps(sgep_seed.PLANTILLAS[0]["preguntas"]))
    assert err is None and len(ok) == 4
    for malo in ["no json", "[]", json.dumps([{"texto": "x", "tipo": "raro"}]),
                 json.dumps([{"texto": "x", "tipo": "opcion", "opciones": ["solo una"]}]),
                 # la condición apunta a una pregunta posterior
                 json.dumps([{"id": "a", "texto": "x", "tipo": "texto", "condicion": {"pregunta": "b", "valor": "si"}},
                             {"id": "b", "texto": "y", "tipo": "si_no"}])]:
        assert sgep.parsear_preguntas(malo)[0] is None


# ── Permisos por rol ─────────────────────────────────────────────────────────

def _login(flask_app, usuario, rol):
    data_store.create_user(usuario, "clave-" + usuario, role=rol)
    c = flask_app.app.test_client()
    assert c.post("/admin/login", data={"username": usuario, "password": "clave-" + usuario}).status_code == 302
    return c


@pytest.fixture
def pm(flask_app):
    return _login(flask_app, "pm_t", "pm")


@pytest.fixture
def auditor(flask_app):
    return _login(flask_app, "aud_t", "auditor")


def _nuevo_proyecto(c):
    sgep.guardar("responsables", [{"id": "r1", "nombre": "R", "email": "r@example.com"}])
    return c.post("/sgep/proyectos/guardar", data={"nombre": "P", "curso": "7mo 1ra", "responsable_id": "r1",
                                                  "inicio": "2026-08-01", "fin": "2026-11-30"})


def test_sin_sesion(client):
    assert client.get("/sgep/").status_code == 302
    assert client.get("/sgep/api/dashboard").status_code == 401


def test_pm_gestiona(pm):
    assert _nuevo_proyecto(pm).status_code == 302
    assert [p["nombre"] for p in sgep.cargar("proyectos")] == ["P"]
    assert pm.get("/sgep/").status_code == 200
    assert pm.get("/sgep/api/dashboard").status_code == 200


def test_auditor_solo_lee(auditor):
    for url in ["/sgep/", "/sgep/proyectos", "/sgep/encuestas", "/sgep/correos", "/sgep/api/dashboard",
                "/sgep/export/excel", "/sgep/export/pdf"]:
        assert auditor.get(url).status_code == 200, url
    assert _nuevo_proyecto(auditor).status_code == 302
    assert sgep.cargar("proyectos") == []


def test_operador_no_entra_al_sgep(operator_client):
    assert operator_client.get("/sgep/").status_code == 302
    assert operator_client.get("/sgep/api/dashboard").status_code == 403


@pytest.mark.parametrize("url", ["/admin/", "/admin/cameras", "/admin/audit", "/admin/users", "/app"])
def test_pm_y_auditor_no_ven_rac(pm, auditor, url):
    for c in (pm, auditor):
        r = c.get(url)
        assert r.status_code == 302 and r.headers["Location"].endswith("/sgep/")


def test_pm_no_opera_rac(pm):
    assert pm.post("/admin/cameras", data={"name": "intrusa"}).status_code == 302
    assert all(c["name"] != "intrusa" for c in data_store.load_cameras())
    assert pm.post("/api/alarm", json={"state": "on"}).status_code == 403


def test_rol_invalido_rechazado(flask_app):
    admin = _login(flask_app, "adm_t", "admin")
    admin.post("/admin/users", data={"action": "create", "username": "x", "password": "clave", "role": "superuser"})
    assert data_store.get_user_by_username("x") is None
    admin.post("/admin/users", data={"action": "create", "username": "y", "password": "clave", "role": "auditor"})
    assert data_store.get_user_by_username("y")["role"] == "auditor"


# ── CSRF ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def csrf_activo(flask_app, monkeypatch):
    monkeypatch.setitem(flask_app.app.config, "WTF_CSRF_ENABLED", True)


def test_csrf_en_formularios_del_sgep(pm, csrf_activo):
    assert _nuevo_proyecto(pm).status_code == 403
    assert sgep.cargar("proyectos") == []
    with pm.session_transaction() as s:
        s["csrf_token"] = "tok"
    r = pm.post("/sgep/proyectos/guardar", data={"csrf_token": "tok", "nombre": "P", "curso": "7mo",
                                                 "responsable_id": "r1", "inicio": "2026-08-01", "fin": "2026-11-30"})
    assert r.status_code == 302 and len(sgep.cargar("proyectos")) == 1


def test_csrf_en_respuesta_publica(client, csrf_activo):
    sgep_seed.sembrar(hoy=HOY)
    inv = next(i for i in sgep.cargar("invitaciones") if not i["respondida"] and i["encuesta_id"].endswith("plsatisf"))
    form = {"nps1": "9", "com1": "4", "ret1": "no"}
    assert client.post(f"/sgep/r/{inv['token']}", data=form).status_code == 403
    client.get(f"/sgep/r/{inv['token']}")  # la página crea el token en la sesión
    with client.session_transaction() as s:
        token = s["csrf_token"]
    assert client.post(f"/sgep/r/{inv['token']}", data=dict(form, csrf_token=token)).status_code == 200


# ── Distribución y respuesta pública ─────────────────────────────────────────

def test_link_unico_responde_una_vez_y_anonimo(client):
    sgep_seed.sembrar(hoy=HOY)
    inv = next(i for i in sgep.cargar("invitaciones") if not i["respondida"] and i["encuesta_id"].endswith("plsatisf"))
    antes = len(sgep.cargar("respuestas"))
    assert client.post(f"/sgep/r/{inv['token']}", data={"nps1": "99"}).status_code == 400
    assert client.post(f"/sgep/r/{inv['token']}", data={"nps1": "10", "com1": "5", "ret1": "no"}).status_code == 200
    nuevas = sgep.cargar("respuestas")[antes:]
    assert len(nuevas) == 1 and nuevas[0]["respondente"] is None      # plantilla anónima
    assert buscar_inv(inv["id"])["respondida"]
    assert "Ya respondiste" in client.post(f"/sgep/r/{inv['token']}", data={"nps1": "1"}).get_data(as_text=True)
    assert len(sgep.cargar("respuestas")) == antes + 1


def buscar_inv(iid):
    return sgep.buscar(sgep.cargar("invitaciones"), iid)


def test_link_general_y_encuesta_cerrada(client, pm):
    sgep_seed.sembrar(hoy=HOY)
    e = next(x for x in sgep.cargar("encuestas") if x["id"].endswith("plavance"))
    form = {"ava1": "50", "rie1": "Tiempo", "con1": "4"}
    assert client.post(f"/sgep/r/{e['link']}", data=form).status_code == 200
    pm.post(f"/sgep/encuestas/{e['id']}/estado")
    assert "Encuesta cerrada" in client.post(f"/sgep/r/{e['link']}", data=form).get_data(as_text=True)
    assert client.get("/sgep/r/token-que-no-existe").status_code == 404


def test_invitar_y_recordar_registra_correos(pm, monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    sgep_seed.sembrar(hoy=HOY)
    eid = "proy0-plsatisf"
    pm.post(f"/sgep/encuestas/{eid}/invitar",
            data={"destinatarios": "Ana <ana@example.com>\nno-es-email\nana@example.com\nbeto@example.com"})
    nuevas = [i for i in sgep.cargar("invitaciones") if i["email"] in ("ana@example.com", "beto@example.com")]
    assert len(nuevas) == 2 and nuevas[0]["nombre"] == "Ana"
    correos = sgep.cargar("correos")
    assert len(correos) == 2 and not correos[0]["enviado"] and nuevas[0]["token"] in correos[1]["cuerpo"]
    pendientes = sum(1 for i in sgep.cargar("invitaciones") if i["encuesta_id"] == eid and not i["respondida"])
    pm.post(f"/sgep/encuestas/{eid}/recordar")
    assert len(sgep.cargar("correos")) == 2 + pendientes


def test_editor_guarda_y_crea_plantilla(pm):
    sgep_seed.sembrar(hoy=HOY)
    preguntas = [{"id": "a1", "texto": "¿Te gustó?", "tipo": "si_no"},
                 {"id": "b1", "texto": "¿Por qué no?", "tipo": "texto", "condicion": {"pregunta": "a1", "valor": "no"}}]
    r = pm.post("/sgep/encuestas/proy1-plsatisf/editar", data={
        "titulo": "Nueva", "tipo": "cierre", "proyecto_id": "proy1", "anonima": "on", "como_plantilla": "on",
        "preguntas_json": json.dumps(preguntas)})
    assert r.status_code == 302
    e = sgep.buscar(sgep.cargar("encuestas"), "proy1-plsatisf")
    assert e["tipo"] == "cierre" and e["preguntas"][1]["condicion"] == {"pregunta": "a1", "valor": "no"}
    assert any(t["titulo"] == "Nueva" for t in sgep.cargar("plantillas"))


def test_exportaciones(auditor):
    sgep_seed.sembrar(hoy=HOY)
    xls = auditor.get("/sgep/export/excel?proyecto=proy0")
    assert xls.status_code == 200 and xls.data[:2] == b"PK"
    pdf = auditor.get("/sgep/export/pdf")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"
