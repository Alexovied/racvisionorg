"""Tests de las mejoras del SGEP tras la auditoría: exportaciones, concurrencia, correos y validaciones."""
import io
import threading
import time
from datetime import date

import pytest

import data_store
import sgep
import sgep_seed

HOY = date(2026, 11, 10)


def _login(flask_app, usuario, rol):
    data_store.create_user(usuario, "clave-" + usuario, role=rol)
    c = flask_app.app.test_client()
    assert c.post("/admin/login", data={"username": usuario, "password": "clave-" + usuario}).status_code == 302
    return c


@pytest.fixture
def pm(flask_app):
    return _login(flask_app, "pm_m", "pm")


@pytest.fixture
def auditor(flask_app):
    return _login(flask_app, "aud_m", "auditor")


def _encuesta_con_texto():
    """Encuesta mínima con una sola pregunta de texto libre; devuelve su link general."""
    sgep.guardar("responsables", [{"id": "r1", "nombre": "R", "email": "r@example.com"}])
    sgep.guardar("proyectos", [{"id": "p1", "nombre": "P", "curso": "7mo", "cliente": "", "responsable_id": "r1",
                                "inicio": "2026-08-01", "fin": "2026-11-30"}])
    sgep.guardar("encuestas", [{"id": "e1", "proyecto_id": "p1", "titulo": "E", "tipo": "lecciones", "anonima": False,
                                "activa": True, "link": "link-general", "creada": "2026-11-01T10:00:00",
                                "preguntas": [{"id": "t1", "texto": "¿Comentarios?", "tipo": "texto",
                                               "opciones": [], "condicion": None}]}])
    return "/sgep/r/link-general"


# ── Exportaciones ────────────────────────────────────────────────────────────

def test_excel_con_texto_libre_malicioso(client, auditor):
    from openpyxl import load_workbook
    url = _encuesta_con_texto()
    for texto in ("=1+1\x0bfin", "- Mejorar la comunicación"):
        assert client.post(url, data={"t1": texto}).status_code == 200
    xls = auditor.get("/sgep/export/excel")
    assert xls.status_code == 200                      # el carácter de control ya no rompe la exportación
    hoja = load_workbook(io.BytesIO(xls.data))["Respuestas"]
    # texto, no fórmula, y sin apóstrofo agregado: en xlsx quedaría a la vista
    assert {(c.data_type, c.value) for c in (hoja["G2"], hoja["G3"])} == {("s", "=1+1fin"), ("s", "- Mejorar la comunicación")}


def test_pdf_con_filtro_con_markup(auditor):
    sgep_seed.sembrar(hoy=HOY)
    pdf = auditor.get("/sgep/export/pdf", query_string={"cliente": "A<B", "x": '<img src="no-existe.png"/>'})
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"


# ── Respuesta pública ────────────────────────────────────────────────────────

def test_respuestas_simultaneas_no_se_pisan(flask_app, monkeypatch):
    url = _encuesta_con_texto()
    cargar = sgep.cargar

    def cargar_lento(nombre):
        datos = cargar(nombre)
        if nombre == "respuestas":
            time.sleep(0.3)  # deja entrar al otro hilo entre leer y guardar
        return datos

    monkeypatch.setattr(sgep, "cargar", cargar_lento)
    hilos = [threading.Thread(target=lambda n=n: flask_app.app.test_client().post(url, data={"t1": f"hilo {n}"}))
             for n in range(2)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert sorted(r["valores"]["t1"] for r in cargar("respuestas")) == ["hilo 0", "hilo 1"]


def test_tope_de_respuestas_por_encuesta(client, monkeypatch):
    url = _encuesta_con_texto()
    monkeypatch.setattr(sgep, "MAX_RESPUESTAS_ENCUESTA", 1)
    assert "Gracias" in client.post(url, data={"t1": "uno"}).get_data(as_text=True)
    assert "Encuesta cerrada" in client.post(url, data={"t1": "dos"}).get_data(as_text=True)
    assert len(sgep.cargar("respuestas")) == 1


def test_encuesta_anonima_guarda_solo_el_dia(client):
    sgep_seed.sembrar(hoy=HOY)
    inv = next(i for i in sgep.cargar("invitaciones") if not i["respondida"] and i["encuesta_id"].endswith("plsatisf"))
    assert client.post(f"/sgep/r/{inv['token']}", data={"nps1": "10", "com1": "5", "ret1": "no"}).status_code == 200
    assert len(sgep.buscar(sgep.cargar("invitaciones"), inv["id"])["respondida"]) == 10   # AAAA-MM-DD, sin hora


# ── Correos e invitaciones ───────────────────────────────────────────────────

def test_auditor_no_ve_links_personales(pm, auditor, monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    sgep_seed.sembrar(hoy=HOY)
    pm.post("/sgep/encuestas/proy0-plsatisf/invitar", data={"destinatarios": "ana@example.com"})
    token = next(i["token"] for i in sgep.cargar("invitaciones") if i["email"] == "ana@example.com")
    assert token in pm.get("/sgep/correos").get_data(as_text=True)
    pagina = auditor.get("/sgep/correos").get_data(as_text=True)
    assert token not in pagina and "/sgep/r/(oculto)" in pagina


def test_invitar_y_recordar_no_pisan_a_quien_responde_durante_el_envio(pm, monkeypatch):
    sgep_seed.sembrar(hoy=HOY)
    eid = "proy0-plsatisf"
    otra = next(i for i in sgep.cargar("invitaciones") if not i["respondida"] and i["encuesta_id"] == eid)
    guardados = []

    def envio_lento(encuesta, proyecto, inv, recordatorio=False):
        """Mientras sale el correo, otra persona responde con su link."""
        lista = sgep.cargar("invitaciones")
        guardados.append(any(i["token"] == inv["token"] for i in lista))
        sgep.buscar(lista, otra["id"])["respondida"] = "2026-11-10"
        sgep.guardar("invitaciones", lista)
        return {"enviado": True, "error": ""}

    monkeypatch.setattr(sgep, "_correo_invitacion", envio_lento)
    pm.post(f"/sgep/encuestas/{eid}/invitar", data={"destinatarios": "ana@example.com"})
    assert guardados == [True]                         # el link se guardó antes de mandarlo
    assert sgep.buscar(sgep.cargar("invitaciones"), otra["id"])["respondida"]

    lista = sgep.cargar("invitaciones")
    sgep.buscar(lista, otra["id"])["respondida"] = None
    sgep.guardar("invitaciones", lista)
    pm.post(f"/sgep/encuestas/{eid}/recordar")
    lista = sgep.cargar("invitaciones")
    assert sgep.buscar(lista, otra["id"])["respondida"]
    assert next(i for i in lista if i["email"] == "ana@example.com")["recordatorios"] == 1


def test_smtp_verifica_el_certificado(monkeypatch):
    import ssl
    from unittest import mock
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.delenv("SMTP_USER", raising=False)
    with mock.patch("smtplib.SMTP") as smtp:
        assert sgep.enviar_correo("a@example.com", "asunto", "cuerpo")["enviado"]
    contexto = smtp.return_value.__enter__.return_value.starttls.call_args.kwargs["context"]
    assert contexto.verify_mode == ssl.CERT_REQUIRED and contexto.check_hostname


def test_invitar_y_recordar_avisan_si_el_smtp_falla(pm, monkeypatch):
    import ssl
    from unittest import mock
    sgep_seed.sembrar(hoy=HOY)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    with mock.patch("smtplib.SMTP", side_effect=ssl.SSLCertVerificationError("certificado autofirmado")):
        for ruta, datos, ok in (("invitar", {"destinatarios": "ana@example.com"}, "0 invitación(es) enviada(s)"),
                                ("recordar", {}, "Recordatorio enviado a 0 persona(s)")):
            pagina = pm.post(f"/sgep/encuestas/proy0-plsatisf/{ruta}", data=datos,
                             follow_redirects=True).get_data(as_text=True)
            assert "no se pudieron enviar" in pagina and ok in pagina   # ya no los da por enviados


# ── Validaciones ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("inicio", ["2026-02-30", "0000-01-01"])
def test_proyecto_con_fecha_imposible_no_se_guarda(pm, inicio):
    sgep.guardar("responsables", [{"id": "r1", "nombre": "R", "email": "r@example.com"}])
    pm.post("/sgep/proyectos/guardar", data={"nombre": "P", "curso": "7mo", "responsable_id": "r1",
                                             "inicio": inicio, "fin": "2026-11-30"})
    assert sgep.cargar("proyectos") == []
    assert pm.get("/sgep/api/dashboard").status_code == 200
