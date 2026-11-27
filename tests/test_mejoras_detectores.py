"""Arreglos de la auditoría de los detectores: redes de YuNet, agachado vs caído, nivel de la
cámara y letterbox de imágenes muy finas."""
import numpy as np
import pytest

import cuerpo
import data_store
import face_detector
import helmet_detector
import motor
import pose

NEGRO = lambda: np.zeros((480, 640, 3), np.uint8)


# ── YuNet con OpenVINO ───────────────────────────────────────────────────────

def test_yunet_openvino_compila_pocas_redes_para_recortes_de_cualquier_tamano():
    class Corte(Exception):
        pass

    pedidas = set()

    def _red(pw, ph):
        pedidas.add((pw, ph))
        raise Corte          # sin modelo: alcanza con ver qué tamaño de red se pide

    yn = object.__new__(face_detector._YuNetOpenVINO)
    yn._red = _red
    for lado in range(30, 260, 10):          # cabezas de 30 a 250 px de ancho
        yn.setInputSize((lado, lado + 10))
        with pytest.raises(Corte):           # y el relleno no puede ser negativo
            yn.detect(np.zeros((lado + 10, lado, 3), np.uint8))
    assert len(pedidas) <= 6                 # con múltiplos de 32 eran 15 redes distintas
    assert all(pw >= 128 and ph >= 128 for pw, ph in pedidas)
    lado = face_detector._YuNetOpenVINO._lado
    assert [lado(n) for n in (30, 140, 150, 266, 285, 600, 800)] == [128, 192, 192, 384, 384, 608, 800]


# ── Hombre caído: agachado o arrodillado ─────────────────────────────────────

def _esqueleto(hombros, caderas, piernas=()):
    """piernas: [(x, y)] de rodilla izq, rodilla der, tobillo izq, tobillo der."""
    p = np.zeros((17, 3), np.float32)
    for i in (5, 6):
        p[i] = (*hombros, 0.9)
    for i in (11, 12):
        p[i] = (*caderas, 0.9)
    for i, xy in zip((13, 14, 15, 16), piernas):
        p[i] = (*xy, 0.9)
    return p


def test_caido_con_las_rodillas_flexionadas_lo_decide_el_torso(monkeypatch):
    # De costado en el borde del cuadro: la caja cortada no alcanza, el torso sí
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    fetal = _esqueleto((60, 300), (240, 310), [(250, 400), (255, 400)])
    assert pose.torso_caido(fetal) is True
    caja = [0, 270, 330, 150]
    assert motor.acostados(NEGRO(), [], [], {"casco": True, "chaleco": False}, [(caja, fetal)]) == [caja]


def test_acostado_con_las_piernas_en_linea_sigue_siendo_caida():
    acostado = _esqueleto((100, 300), (300, 320), [(420, 330), (420, 335), (540, 335), (540, 340)])
    assert pose.torso_caido(acostado) is True
    assert pose.torso_caido(_esqueleto((100, 300), (300, 320))) is True      # sin piernas a la vista


def test_caido_encogido_lo_sigue_viendo_la_forma_de_la_caja(monkeypatch):
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    encogido = _esqueleto((150, 300), (330, 310), [(340, 400), (345, 400)])  # de costado, rodillas hacia abajo
    caja = [100, 270, 300, 150]                           # ancha y entera dentro de la imagen
    assert motor.acostados(NEGRO(), [], [], {"casco": True, "chaleco": False}, [(caja, encogido)]) == [caja]


# ── Inclinación de la cámara ─────────────────────────────────────────────────

def test_nivel_de_la_camara_no_gira_por_alguien_agachado_unos_segundos():
    cam = data_store.create_camera("Obra", "Sector A", "usb", "0")
    n = motor.Camara(cam).nivel
    for _ in range(10):
        n.actualizar(25.0)                   # 10 frames de alguien arrodillado
    assert not n.estable()                   # antes ya alcanzaba para girar toda la imagen
    n = cuerpo.Nivel(n=120, cada=10)
    for a in [0.0] * 100 + [25.0] * 30:      # cámara derecha y después alguien agachado
        n.actualizar(a)
    assert n.estable() and n.actualizar(None) == pytest.approx(0)
    # Un giro equivocado (alguien solo e inclinado mucho rato, o la cámara movida) no dura horas
    n = motor.Camara(cam).nivel
    for _ in range(400):
        n.actualizar(25.0)
    medidas = 0
    while n.actualizar(0.0) >= cuerpo.INCLINACION_MIN:
        medidas += 1
    assert medidas <= 250


# ── Letterbox ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("forma", [(1, 2000, 3), (2000, 1, 3)])
def test_letterbox_de_una_imagen_muy_fina_no_falla(forma):
    canvas, *_ = helmet_detector._letterbox(np.zeros(forma, np.uint8))
    assert canvas.shape == (640, 640, 3)
