"""Mini benchmark: tiempo de operaciones clave de RAC Vision (se corre dentro de pytest para reutilizar las fixtures)."""
import json, time, statistics, os
import numpy as np
from unittest import mock
import data_store, epp

def _medir(fn, n):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    return round(statistics.mean(t), 3), round(statistics.median(t), 3), round(max(t), 3)

def test_benchmark(operator_client, tmp_data):
    res = {}
    with mock.patch("admin.enviar_comando_esp32"):
        res["POST /admin/api/events (vacío→200 eventos)"] = _medir(
            lambda: operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80}), 200)
    eventos = [{"id": f"e{i}", "type": "con_casco" if i % 2 else "sin_casco", "confidence": 80.0,
                "camera": "webcam", "timestamp": "2026-11-10T10:00:00"} for i in range(5000)]
    (tmp_data / "events.json").write_text(json.dumps(eventos))
    res["get_filtered_events (5000 eventos, filtro tipo+fecha)"] = _medir(
        lambda: data_store.get_filtered_events(event_type="sin_casco", date_from="2026-11-01", page=1, per_page=25), 30)
    with mock.patch("admin.enviar_comando_esp32"):
        res["POST /admin/api/events (con 5000 eventos guardados)"] = _medir(
            lambda: operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80}), 30)
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    res["epp.tiene_chaleco (frame 640x480, regla de color)"] = _medir(
        lambda: epp.tiene_chaleco(img, [270, 60, 100, 100]), 300)
    open(os.environ.get("BENCH_OUT", "bench.json"), "w").write(json.dumps(res, ensure_ascii=False, indent=1))
