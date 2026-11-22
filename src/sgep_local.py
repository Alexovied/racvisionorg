"""Corre solo el SGEP-Dashboard en localhost, sin TensorFlow, OpenVINO ni ESP32.

Uso:
    pip install -r requirements-sgep.txt
    python src/sgep_local.py            → http://localhost:5000

Usa su propia carpeta de datos (data_sgep/) y la llena con datos de prueba la
primera vez. Usuarios de prueba (solo para uso local):
    admin / admin-demo-2026 · pm_demo / pm-demo-2026 · auditor_demo / auditor-demo-2026
"""
import os
import secrets
import sys
from unittest import mock

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

# El panel de RAC (admin.py) importa visión por computadora y el ESP32; el SGEP no los usa.
# Los reemplazos se quitan después de importar admin: si numpy quedara simulado,
# openpyxl lo detectaría como instalado y la exportación a Excel fallaría.
sys.modules["face_id"] = mock.MagicMock()
sys.modules["hardware"] = mock.MagicMock()
_simulados = []
for _modulo in ("cv2", "numpy", "requests"):
    try:
        __import__(_modulo)
    except ImportError:
        sys.modules[_modulo] = mock.MagicMock()
        _simulados.append(_modulo)

os.environ.setdefault("ADMIN_PASSWORD", "admin-demo-2026")
os.environ.setdefault("N8N_WEBHOOK_URL", "")

from flask import Flask, abort, redirect, request, send_from_directory, session, url_for  # noqa: E402

import data_store  # noqa: E402

data_store.DATA_DIR = os.environ.get("SGEP_DATA_DIR") or os.path.join(RAIZ, "data_sgep")

import admin  # noqa: E402

for _modulo in _simulados:
    del sys.modules[_modulo]

import sgep  # noqa: E402
import sgep_seed  # noqa: E402

app = Flask(__name__, root_path=RAIZ)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.register_blueprint(admin.admin_bp)
app.register_blueprint(sgep.sgep_bp)
admin.init_login(app)
app.jinja_env.globals["solo_sgep"] = True


# CSRF: misma regla que app.py (campo csrf_token o header X-CSRFToken en todo POST)
def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def verificar_csrf():
    if request.method != "POST" or not app.config.get("WTF_CSRF_ENABLED", True):
        return None
    enviado = request.headers.get("X-CSRFToken") or request.form.get("csrf_token", "")
    esperado = session.get("csrf_token")
    if not esperado or not secrets.compare_digest(enviado, esperado):
        abort(403, "Token CSRF inválido o ausente. Recargá la página e intentá de nuevo.")
    return None


# Sin RAC: del panel admin solo quedan login, logout y usuarios
ADMIN_PERMITIDOS = {"admin.login", "admin.logout", "admin.users", "admin.delete_user"}


@app.before_request
def solo_sgep():
    ep = request.endpoint or ""
    if ep.startswith("admin.") and ep not in ADMIN_PERMITIDOS:
        return redirect(url_for("sgep.dashboard"))
    return None


@app.route("/")
def inicio():
    return redirect(url_for("sgep.dashboard"))


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(RAIZ, "static"), "favicon.ico")


if not sgep.cargar("proyectos"):
    print("Cargando datos de prueba:", sgep_seed.sembrar())

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", "5000"))
    print(f"SGEP en http://localhost:{puerto}  (admin / {os.environ['ADMIN_PASSWORD']}, "
          "pm_demo / pm-demo-2026, auditor_demo / auditor-demo-2026)")
    app.run(host="127.0.0.1", port=puerto)
