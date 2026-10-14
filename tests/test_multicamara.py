"""Varias cámaras: /app/multi y eventos de webcam-2."""


def test_multi_muestra_una_instancia_por_camara(operator_client):
    html = operator_client.get("/app/multi").get_data(as_text=True)
    assert 'src="/app?cam=1"' in html and 'src="/app?cam=2"' in html
    assert 'allow="camera"' in html


def test_evento_de_la_segunda_camara_se_acepta(operator_client):
    r = operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80, "camera": "webcam-2"})
    assert r.status_code == 200


def test_camara_inventada_se_rechaza(operator_client):
    for cam in ("webcam-x", "webcam-", "otra"):
        r = operator_client.post("/admin/api/events", json={"type": "sin_casco", "confidence": 80, "camera": cam})
        assert r.status_code == 400, cam
