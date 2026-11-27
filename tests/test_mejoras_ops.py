"""Mejoras de operación: comando de arranque, .env.example y scripts de herramientas/."""
import os
import re
import sys
import zipfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "herramientas"))


def _leer(nombre):
    with open(os.path.join(RAIZ, nombre), encoding="utf-8") as f:
        return f.read()


def test_gunicorn_arranca_con_un_proceso_y_varios_hilos():
    """El estado (motor, wearables, candado de data_store) vive en el proceso: un solo worker,
    y con hilos para que un video MJPEG abierto no deje afuera al resto de los pedidos."""
    for nombre in ("README.md",):
        comando = re.search(r"gunicorn .*", _leer(nombre)).group(0)
        assert "--workers 1 " in comando and re.search(r"--threads \d\d", comando), nombre


def test_env_example_no_trae_credenciales_que_sirvan():
    """Quien copie .env.example sin editarlo no puede quedar con una contraseña o clave conocida."""
    env = _leer(".env.example")
    assert re.search(r"^ADMIN_PASSWORD=\s*$", env, re.M) and re.search(r"^SECRET_KEY=\s*$", env, re.M)


def test_armar_zip_con_una_sola_clase(tmp_path, monkeypatch):
    import capturar_casco
    monkeypatch.setattr(capturar_casco, "RAIZ", str(tmp_path))
    (tmp_path / "entrenamiento").mkdir()
    (tmp_path / "datasets" / "con_casco").mkdir(parents=True)   # falta sin_casco
    (tmp_path / "datasets" / "con_casco" / "a.jpg").write_bytes(b"jpg")
    capturar_casco.armar_zip()
    with zipfile.ZipFile(tmp_path / "entrenamiento" / "fotos_rac.zip") as z:
        assert z.namelist() == ["con_casco/a.jpg"]


def test_benchmark_de_carga_encuentra_el_token_csrf_del_login(client):
    import benchmark_carga
    assert benchmark_carga.RE_CSRF.search(client.get("/admin/login").get_data(as_text=True)).group(1)
