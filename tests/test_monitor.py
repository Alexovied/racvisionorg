"""Monitoreo de varias cámaras desde el panel admin."""
import data_store


def test_monitor_muestra_una_tarjeta_por_camara_activa(operator_client):
    a = data_store.create_camera("Entrada", "Planta baja", "webcam")
    b = data_store.create_camera("Taller", "", "webcam")
    apagada = data_store.create_camera("Depósito", "", "webcam")
    data_store.toggle_camera(apagada["id"])

    html = operator_client.get("/admin/monitor").get_data(as_text=True)
    assert f'data-cam-id="{a["id"]}"' in html and f'data-cam-id="{b["id"]}"' in html
    assert apagada["id"] not in html


def test_monitor_sin_camaras_invita_a_registrarlas(operator_client):
    html = operator_client.get("/admin/monitor").get_data(as_text=True)
    assert "No hay cámaras activas" in html


def test_evento_con_id_de_camara_registrada_se_acepta(operator_client):
    cam = data_store.create_camera("Entrada", "", "webcam")
    r = operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80, "camera": cam["id"]})
    assert r.status_code == 200
