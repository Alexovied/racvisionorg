"""cuerpo.py: cabezas que tienen un cuerpo (casco sobre la mesa ≠ persona) y cámara girada."""
import cv2
import numpy as np
import pytest

import cuerpo
import motor


def _esqueleto(cara=(), hombros=(), caderas=(), conf=0.9):
    """17 puntos COCO: cara = [(x, y)] para nariz, ojo izq, ojo der, oreja izq, oreja der."""
    p = np.zeros((17, 3), np.float32)
    for i, xy in enumerate(cara):
        p[i] = (*xy, conf)
    for i, xy in zip((5, 6), hombros):
        p[i] = (*xy, conf)
    for i, xy in zip((11, 12), caderas):
        p[i] = (*xy, conf)
    return p


DE_FRENTE = ([0, 0, 1, 1], _esqueleto(cara=[(320, 140), (305, 130), (335, 130), (290, 135), (350, 135)],
                                      hombros=[(380, 220), (260, 220)], caderas=[(360, 400), (280, 400)]))
CABEZA = [290, 100, 60, 70]


def test_cabeza_con_cara_y_hombros_es_persona():
    assert cuerpo.es_persona(CABEZA, [DE_FRENTE], 480)


def test_casco_sin_cuerpo_no_es_persona():
    assert not cuerpo.es_persona(CABEZA, [], 480)
    # Un casco sobre la mesa al lado de alguien: el esqueleto existe, pero no es de esa cabeza
    assert not cuerpo.es_persona([520, 300, 60, 50], [DE_FRENTE], 480)


def test_casco_a_la_altura_del_hombro_no_es_persona():
    # Apoyado en un escritorio, justo al costado del hombro de la persona sentada
    assert not cuerpo.es_persona([395, 200, 50, 40], [DE_FRENTE], 480)


def test_de_espaldas_alcanzan_los_dos_hombros_debajo():
    espalda = ([0, 0, 1, 1], _esqueleto(hombros=[(260, 220), (380, 220)]))
    assert cuerpo.es_persona(CABEZA, [espalda], 480)
    assert not cuerpo.es_persona([600, 100, 60, 70], [espalda], 480)


def test_cara_muy_cerca_sin_hombros_en_la_imagen():
    cerca = ([0, 0, 1, 1], _esqueleto(cara=[(320, 300), (280, 250), (360, 250)]))
    assert cuerpo.es_persona([180, 100, 280, 360], [cerca], 480)


@pytest.mark.parametrize("giro", [-30, -15, 0, 20, 35])
def test_inclinacion_mide_el_giro_del_torso(giro):
    # Persona derecha girada `giro` grados en sentido horario alrededor de (320, 240)
    M = cv2.getRotationMatrix2D((320, 240), -giro, 1.0)
    caja, kp = cuerpo.mover_cuerpo(DE_FRENTE, M)
    assert cuerpo.inclinacion([(caja, kp)]) == pytest.approx(giro, abs=0.5)


def test_inclinacion_ignora_a_los_caidos_y_usa_la_mediana():
    acostado = ([0, 0, 1, 1], _esqueleto(hombros=[(100, 300), (100, 340)], caderas=[(300, 300), (300, 340)]))
    assert cuerpo.inclinacion([acostado]) is None
    assert cuerpo.inclinacion([DE_FRENTE, DE_FRENTE, acostado]) == pytest.approx(0, abs=0.5)


def test_nivel_promedia_y_recuerda():
    n = cuerpo.Nivel()
    assert n.actualizar(20) == 20
    for a in (18, 22, 40, 19):
        n.actualizar(a)
    assert n.actualizar(None) == pytest.approx(20)     # sin gente: sigue la última medida


def test_enderezar_y_volver_conservan_la_posicion():
    img = np.zeros((480, 640, 3), np.uint8)
    derecha, M, M_inv = cuerpo.enderezar(img, 25)
    assert derecha.shape[0] > 480 and derecha.shape[1] > 640       # no se recortan las esquinas
    x, y = cuerpo.mover(cuerpo.mover((100, 50), M), M_inv)
    assert (x, y) == pytest.approx((100, 50), abs=1e-3)
    cx, cy = cuerpo.mover((320, 240), M)
    assert cuerpo.volver_cabeza([cx - 20, cy - 20, 40, 40], M_inv, img.shape) == [300, 220, 40, 40]


# ── En el motor ──────────────────────────────────────────────────────────────

@pytest.fixture
def modelos(monkeypatch):
    """Pose y cabezas simulados: una persona real y un casco suelto."""
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    monkeypatch.setattr(motor.face_id, "identificar", lambda *a, **k: None)
    casco_suelto = {"bbox": [520, 330, 60, 50], "clase": "casco", "conf": 0.9, "p_casco": 0.9, "p_sin": 0.1}
    persona = {"bbox": CABEZA, "clase": "sin_casco", "conf": 0.9, "p_casco": 0.1, "p_sin": 0.9}
    monkeypatch.setattr(motor.helmet_detector, "detectar", lambda img: [persona, casco_suelto])
    monkeypatch.setattr(motor.pose, "detectar", lambda img: [DE_FRENTE])


def test_motor_descarta_el_casco_apoyado(modelos):
    caras, _ = motor.analizar(np.zeros((480, 640, 3), np.uint8), {"casco": True, "chaleco": False})
    assert [c["bbox"] for c in caras] == [CABEZA]
    assert caras[0]["sin_casco"] == pytest.approx(90)


def test_motor_sin_modelo_de_pose_no_filtra(modelos, monkeypatch):
    monkeypatch.setattr(motor.pose, "disponible", lambda: False)
    caras, _ = motor.analizar(np.zeros((480, 640, 3), np.uint8), {"casco": True, "chaleco": False})
    assert len(caras) == 2


def test_motor_endereza_la_camara_girada(monkeypatch):
    monkeypatch.setattr(motor.pose, "disponible", lambda: True)
    monkeypatch.setattr(motor.face_id, "identificar", lambda *a, **k: None)
    M = cv2.getRotationMatrix2D((320, 240), -20, 1.0)                 # cámara girada 20° horario
    girado = cuerpo.mover_cuerpo(DE_FRENTE, M)
    monkeypatch.setattr(motor.pose, "detectar", lambda img: [girado])
    vistas = []

    def cabezas(img):
        vistas.append(img.shape)
        if img.shape[0] == 480:
            return []                    # con la cámara girada el detector no ve la cabeza
        # El detector recibe la imagen enderezada: la cabeza está donde estaría sin giro
        _, M2, _ = cuerpo.enderezar(np.zeros((480, 640, 3), np.uint8), 20)
        cx, cy = cuerpo.mover(cuerpo.mover((320, 135), M), M2)
        return [{"bbox": [int(cx - 30), int(cy - 35), 60, 70], "clase": "casco", "conf": 0.9,
                 "p_casco": 0.9, "p_sin": 0.1}]
    monkeypatch.setattr(motor.helmet_detector, "detectar", cabezas)
    info = {}
    caras, _ = motor.analizar(np.zeros((480, 640, 3), np.uint8), {"casco": True, "chaleco": False}, info=info)
    assert info["inclinacion"] == pytest.approx(20, abs=0.5)
    assert vistas[0][0] == 480 and vistas[1][0] > 480                  # sin cabezas, prueba enderezada
    assert len(caras) == 1
    x, y, w, h = caras[0]["bbox"]
    assert (x + w / 2, y + h / 2) == pytest.approx(cuerpo.mover((320, 135), M), abs=2)   # vuelve a la original


def test_giro_de_un_solo_frame_no_endereza_si_ya_ve_la_cabeza(modelos, monkeypatch):
    # Alguien que se recuesta o ladea la cabeza cambia el giro medido: si la imagen tal cual ya
    # encuentra la cabeza, no se endereza (enderezar de más confunde pelo con casco).
    M = cv2.getRotationMatrix2D((320, 240), -20, 1.0)
    monkeypatch.setattr(motor.pose, "detectar", lambda img: [DE_FRENTE, cuerpo.mover_cuerpo(DE_FRENTE, M)])
    monkeypatch.setattr(cuerpo, "inclinacion", lambda cuerpos: 20.0)
    vistas = []
    original = motor.helmet_detector.detectar
    monkeypatch.setattr(motor.helmet_detector, "detectar", lambda img: vistas.append(img.shape) or original(img))
    caras, _ = motor.analizar(np.zeros((480, 640, 3), np.uint8), {"casco": True, "chaleco": False})
    assert [c["bbox"] for c in caras] == [CABEZA] and vistas == [(480, 640, 3)]


def test_camara_fija_con_giro_estable_endereza_siempre(modelos, monkeypatch):
    monkeypatch.setattr(cuerpo, "inclinacion", lambda cuerpos: 20.0)
    vistas = []
    original = motor.helmet_detector.detectar
    monkeypatch.setattr(motor.helmet_detector, "detectar", lambda img: vistas.append(img.shape) or original(img))
    nivel = cuerpo.Nivel()
    for _ in range(10):
        nivel.actualizar(20.0)
    motor.analizar(np.zeros((480, 640, 3), np.uint8), {"casco": True, "chaleco": False}, nivel=nivel)
    assert len(vistas) == 1 and vistas[0][0] > 480
