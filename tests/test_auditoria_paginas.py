"""Paginación de Auditoría: 1 … 5 6 7 … 21."""
import re

import data_store


def test_paginacion_muestra_extremos_vecinos_y_un_salto_por_lado(operator_client):
    for _ in range(21 * 25):
        data_store.append_event("con_casco", 90)
    html = operator_client.get("/admin/audit?page=6").get_data(as_text=True)
    nav = html[html.index('class="pagination"'):]
    nav = nav[:nav.index("</div>")]
    items = re.findall(r'page-btn[^>]*>\s*(\d+)\s*<|page-dots">(…)<', nav)
    assert [a or b for a, b in items] == ["1", "…", "5", "6", "7", "…", "21"]
