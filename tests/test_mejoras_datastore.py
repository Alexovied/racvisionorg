"""Mejoras de data_store.py y face_id.py: archivos JSON dañados, fsync y fotos huérfanas."""
import json
import os

import numpy as np
import pytest

import data_store
import face_id


def test_guardar_json_hace_fsync_antes_de_reemplazar(tmp_data, monkeypatch):
    orden = []
    fsync, replace = os.fsync, os.replace
    monkeypatch.setattr(data_store.os, "fsync", lambda fd: (orden.append("fsync"), fsync(fd))[1])
    monkeypatch.setattr(data_store.os, "replace", lambda a, b: (orden.append("replace"), replace(a, b))[1])
    data_store.save_cameras([{"id": "c1"}])
    assert orden == ["fsync", "replace"]
    assert data_store.load_cameras() == [{"id": "c1"}]


def test_events_json_se_guarda_compacto_y_el_resto_con_sangria(tmp_data):
    data_store.append_event("sin_casco", 90, persona="José")
    data_store.save_cameras([{"id": "c1"}])
    texto = (tmp_data / "events.json").read_text(encoding="utf-8")
    assert "\n" not in texto and json.loads(texto)[0]["persona"] == "José"
    assert "\n" in (tmp_data / "cameras.json").read_text(encoding="utf-8")


def test_events_json_dañado_se_aparta_y_se_sigue_registrando(tmp_data, capsys):
    tmp_data.mkdir(exist_ok=True)
    (tmp_data / "events.json").write_text('[{"id": "x", "type": "sin_ca')
    evento = data_store.append_event("sin_casco", 90)
    assert [e["id"] for e in data_store.load_events()] == [evento["id"]]
    apartados = list(tmp_data.glob("events.json.corrupto-*"))
    assert len(apartados) == 1 and apartados[0].read_text() == '[{"id": "x", "type": "sin_ca'
    assert "events.json" in capsys.readouterr().err


def test_foto_que_no_se_pudo_escribir_no_queda_huerfana(tmp_data):
    # Un str en vez de bytes hace fallar f.write con el archivo ya creado, como un disco lleno
    with pytest.raises(TypeError):
        data_store.append_event("sin_casco", 90, foto_webp="no son bytes")
    assert list((tmp_data / "alertas").glob("*")) == []
    assert data_store.load_events() == []


def test_personas_json_dañado_no_tumba_la_deteccion(tmp_data, monkeypatch):
    monkeypatch.setitem(face_id._cache, "mtime", None)
    tmp_data.mkdir(exist_ok=True)
    (tmp_data / "personas.json").write_text('{"alex": [[0.1, 0.')
    img = np.zeros((100, 100, 3), np.uint8)
    assert face_id.identificar(img, [10, 10, 40, 40]) is None
    # y no se relee en cada cuadro: el fallo queda cacheado hasta que cambie el archivo
    monkeypatch.setattr(face_id, "_leer_db", lambda: pytest.fail("releyó personas.json"))
    assert face_id.identificar(img, [10, 10, 40, 40]) is None
