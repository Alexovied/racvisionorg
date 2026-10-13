"""Reconocimiento de personas (face_id.py): comparación contra data/personas.json."""
import json

import numpy as np

import face_id


def _unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _alta(tmp_data, personas):
    tmp_data.mkdir(exist_ok=True)
    (tmp_data / "personas.json").write_text(json.dumps({n: [e.tolist() for e in embs] for n, embs in personas.items()}))


def test_identifica_a_la_persona_mas_parecida(tmp_data, monkeypatch):
    _alta(tmp_data, {"alex": [_unit(1, 0, 0)], "mateo": [_unit(0, 1, 0)]})
    monkeypatch.setattr(face_id, "embedding", lambda img: _unit(0.1, 1, 0))
    img = np.zeros((100, 100, 3), np.uint8)
    assert face_id.identificar(img, [10, 10, 40, 40]) == "mateo"


def test_cara_que_no_se_parece_a_nadie_es_desconocido(tmp_data, monkeypatch):
    _alta(tmp_data, {"alex": [_unit(1, 0, 0)]})
    monkeypatch.setattr(face_id, "embedding", lambda img: _unit(0, 0, 1))
    assert face_id.identificar(np.zeros((100, 100, 3), np.uint8), [10, 10, 40, 40]) == "Desconocido"


def test_sin_cara_visible_devuelve_none(tmp_data, monkeypatch):
    _alta(tmp_data, {"alex": [_unit(1, 0, 0)]})
    monkeypatch.setattr(face_id, "embedding", lambda img: None)
    assert face_id.identificar(np.zeros((100, 100, 3), np.uint8), [10, 10, 40, 40]) is None


def test_sin_personas_dadas_de_alta_devuelve_none(tmp_data):
    assert face_id.identificar(np.zeros((100, 100, 3), np.uint8), [10, 10, 40, 40]) is None
