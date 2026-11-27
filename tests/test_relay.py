"""herramientas/relay_camara.py: el relay corre sin supervisión en la PC del cliente y no puede
morir en silencio ni martillar al servidor cuando este lo rechaza."""
import os
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "herramientas"))
import relay_camara  # noqa: E402


def _reloj(monkeypatch, dormir):
    """Reemplaza time solo dentro del relay (el time.sleep real lo usan hilos de otros tests)."""
    monkeypatch.setattr(relay_camara, "time", SimpleNamespace(sleep=dormir, monotonic=time.monotonic))


def _un_envio(monkeypatch, post):
    """Corre enviar() con un cuadro nuevo hasta la primera espera y devuelve cuánto esperó."""
    esperas = []

    def dormir(s):
        esperas.append(s)
        raise SystemExit  # corta el while True del emisor

    _reloj(monkeypatch, dormir)
    lector = SimpleNamespace(cuadro=np.zeros((48, 64, 3), np.uint8))
    with pytest.raises(SystemExit):
        relay_camara.enviar(SimpleNamespace(post=post), "http://x/cuadro", lector,
                            SimpleNamespace(ancho=960, hilos=1, fps=8),
                            {"n": 0, "enviados": 0, "ultimo": None}, threading.Lock())
    return esperas


def test_lector_sobrevive_a_un_error_de_opencv(monkeypatch):
    aperturas = []

    def abrir(*a):
        aperturas.append(a)
        raise cv2.error("driver roto") if len(aperturas) == 1 else SystemExit

    monkeypatch.setattr(relay_camara.cv2, "VideoCapture", abrir)
    _reloj(monkeypatch, lambda s: None)
    with pytest.raises(SystemExit):
        relay_camara.Lector("rtsp://x").run()
    assert len(aperturas) == 2  # después del error volvió a intentar


def test_emisor_espera_si_el_servidor_rechaza_el_cuadro(monkeypatch):
    rechazo = SimpleNamespace(status_code=409, ok=False, text="Cámara desactivada")
    assert _un_envio(monkeypatch, lambda *a, **k: rechazo) == [2]


def test_emisor_no_manda_un_post_vacio_si_no_pudo_codificar(monkeypatch):
    monkeypatch.setattr(relay_camara, "jpeg", lambda img, ancho: None)

    def post(*a, **k):
        raise AssertionError("mandó un POST sin cuerpo")

    assert _un_envio(monkeypatch, post) == [2]


def test_main_sale_si_se_cae_un_hilo(monkeypatch):
    class HiloMuerto:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

    vueltas = []

    def dormir(s):
        vueltas.append(s)
        assert len(vueltas) < 3, "siguió corriendo sin hilos"

    def salir(codigo):
        raise SystemExit(codigo)

    monkeypatch.setattr(relay_camara, "Lector", HiloMuerto)
    monkeypatch.setattr(relay_camara, "threading", SimpleNamespace(Thread=HiloMuerto, Lock=threading.Lock))
    monkeypatch.setattr(relay_camara, "os", SimpleNamespace(environ={}, _exit=salir))
    _reloj(monkeypatch, dormir)
    monkeypatch.setattr(sys, "argv", ["relay", "--camara", "rtsp://x", "--servidor", "http://x", "--id", "1",
                                      "--token", "t"])
    with pytest.raises(SystemExit) as e:
        relay_camara.main()
    assert e.value.code == 1
