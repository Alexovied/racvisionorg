# SGEP-Dashboard — Manual técnico

## Arquitectura

El SGEP es un **blueprint de Flask** (`sgep_bp`, prefijo `/sgep`) registrado en `app.py` junto al panel de RAC Vision. Reutiliza:

| Pieza | De dónde | Para qué |
|-------|----------|----------|
| Login y sesión | `admin.py` (`init_login`, flask-login) | Mismo `/admin/login` para todos los roles |
| Permisos | `admin.roles_required` | `LECTURA = admin, pm, auditor`; `GESTION = admin, pm` |
| CSRF | `app.verificar_csrf` (`before_request` global) | Todo POST, incluida la encuesta pública, exige `csrf_token` |
| Persistencia | `data_store._load_json/_save_json` | Archivos `data/sgep_*.json` |
| Estilos | `static/admin.css` + `static/sgep.css` | Mismo look del panel; `sgep.css` agrega editor, semáforo y mapa de calor |
| Accesibilidad | `static/accesibilidad.{css,js}` | También en la página pública de respuesta |

```
app.py ──registra──► admin_bp (/admin)      ◄── login, usuarios, RAC
       └─registra──► sgep_bp  (/sgep)       ◄── sgep.py
sgep.py ──usa──► data_store (JSON en data/)
         ──usa──► admin.roles_required, admin.is_api_request
templates/sgep/*.html  (extienden admin/base.html, salvo responder.html que es pública)
sgep_seed.py           datos de prueba
tests/test_sgep.py     tests
```

### Aislamiento de roles

`pm` y `auditor` son roles exclusivos del SGEP. `sgep.confinar_roles_sgep` (`before_app_request`) los redirige a `/sgep/` si intentan entrar a cualquier endpoint `admin.*` (salvo login/logout) o a `/app`, `/predict`, `/reset_buffer`, `/api/alarm`; en rutas de API responde 403 JSON. El sidebar de `admin/base.html` oculta el menú de RAC para esos roles y muestra el del SGEP a `admin`, `pm` y `auditor`. `admin.users()` ahora valida el rol contra `admin.ROLES`.

## Modelo de datos (`data/sgep_*.json`)

| Archivo | Campos principales |
|---------|-------------------|
| `responsables` | `id, nombre, email` |
| `proyectos` | `id, nombre, curso, cliente, responsable_id, inicio, fin, descripcion, creado` |
| `plantillas` | `id, titulo, tipo, anonima, preguntas[]` |
| `encuestas` | `id, proyecto_id, titulo, tipo, anonima, preguntas[], activa, link, creada` |
| `invitaciones` | `id, encuesta_id, nombre, email, token, creada, recordatorios, respondida` |
| `respuestas` | `id, encuesta_id, proyecto_id, fecha, respondente (None si anónima), valores{pregunta_id: valor}` |
| `correos` | `id, fecha, para, asunto, cuerpo, enviado, error` (últimos 500) |

Pregunta: `{id, texto, tipo, opciones[], condicion: {pregunta, valor} | null}`; `tipo` ∈ `nps, avance, escala, si_no, opcion, texto`.

`tipo` de encuesta ∈ `satisfaccion, avance, lecciones, cierre`.

## Rutas

| Método | Ruta | Rol | Qué hace |
|--------|------|-----|----------|
| GET | `/sgep/` | lectura | Dashboard |
| GET | `/sgep/api/dashboard?proyecto&cliente&responsable&tipo&desde&hasta` | lectura | KPIs y series en JSON (lo consulta el dashboard cada 10 s) |
| GET | `/sgep/proyectos` | lectura | Listado de proyectos y responsables |
| POST | `/sgep/proyectos/guardar`, `/sgep/proyectos/<id>/borrar` | gestión | Alta/edición/baja de proyectos |
| POST | `/sgep/responsables/guardar`, `/sgep/responsables/<id>/borrar` | gestión | Alta/edición/baja de responsables |
| GET | `/sgep/encuestas` | lectura | Encuestas y plantillas |
| POST | `/sgep/encuestas/nueva` | gestión | Crea desde plantilla o en blanco |
| GET/POST | `/sgep/encuestas/<id>/editar` | gestión | Constructor drag & drop |
| POST | `/sgep/encuestas/<id>/borrar`, `/sgep/plantillas/<id>/borrar` | gestión | Bajas (la encuesta arrastra invitaciones y respuestas) |
| GET | `/sgep/encuestas/<id>` | lectura | Distribución: link, QR, control de respuestas |
| POST | `/sgep/encuestas/<id>/invitar`, `/recordar`, `/estado` | gestión | Invitar por email, recordar, abrir/cerrar |
| GET | `/sgep/correos` | lectura | Registro de correos |
| GET/POST | `/sgep/r/<token>` | pública | Responder (token de invitación o `link` general) |
| GET | `/sgep/export/excel`, `/sgep/export/pdf` | lectura | Reportes con los filtros del query string |

## Cálculo de KPIs (`sgep.py`, funciones puras)

- `nps(puntajes)` = `round((#≥9 − #≤6) · 100 / n)`; `None` sin datos.
- `avance_esperado(proyecto, día)` = % del plazo `inicio → fin` transcurrido, acotado a 0–100.
- `cumplimiento(percibido, esperado)` = `min(100, percibido / esperado · 100)`. En el dashboard se calcula **por respuesta** contra lo esperado el día en que se respondió y se promedia por proyecto.
- Avance percibido de un proyecto = su respuesta de avance más reciente; el KPI global promedia los proyectos.
- `participacion(invitaciones)` = respondidas / total · 100.
- `semaforo(nps, cumplimiento)`: rojo si NPS < 0 o cumplimiento < 60; amarillo si NPS < 30 o cumplimiento < 85; verde si no; gris sin datos.
- `calcular_dashboard(filtros)` arma KPIs, filas por proyecto, torta, tendencia semanal (semanas ISO desde el lunes), mapa de calor y últimos comentarios de texto.

## Validación y seguridad

- **Editor** (`parsear_preguntas`): máx. 50 preguntas, texto ≤ 300, tipo válido, opción múltiple con ≥ 2 opciones, la condición solo puede apuntar a una pregunta **anterior** (no hay ciclos).
- **Respuesta pública** (`validar_respuesta`): recorre las preguntas en orden, ignora las ocultas por la lógica condicional (aunque el navegador las mande), valida rangos (NPS 0–10, avance 0–100, escala 1–5), sí/no y opciones; texto ≤ 1000. Error → 400 y se conserva lo ya completado.
- **Link único**: `secrets.token_urlsafe(16)` por invitación; una sola respuesta por token. El link general (QR) admite varias respuestas (decisión consciente para el stand).
- **Anonimato**: con `anonima=True` la respuesta guarda `respondente=None`; la invitación igual se marca como respondida para medir participación.
- **CSRF**: el `before_request` global de `app.py` cubre todos los POST del SGEP; las plantillas incluyen el campo `csrf_token`.
- **XSS**: Jinja escapa todo; el JS del dashboard y del editor escapa los textos que inserta con `esc()`; los datos van al JS con `|tojson`.
- **Emails**: se validan con regex antes de usarlos; `EmailMessage` rechaza saltos de línea en cabeceras.

## Correo

`enviar_correo()` usa SMTP con STARTTLS si existe `SMTP_HOST` (`SMTP_PORT`=587, `SMTP_USER`, `SMTP_PASS`, `SMTP_FROM`). Sin `SMTP_HOST` solo registra el correo (modo demo). Siempre queda en `sgep_correos.json`. Los recordatorios son manuales (botón); para automatizarlos habría que agregar un job programado que llame a la misma lógica.

## Dependencias nuevas

`openpyxl` (Excel) y `reportlab` (PDF), agregadas a `requirements.txt`. En el navegador: Chart.js 4.4.0 (ya usado por el dashboard de RAC) y `qrcode-generator` 1.4.4 desde cdnjs para el QR.

## Correr localmente

```bash
pip install -r requirements-dev.txt
python sgep_seed.py          # datos de prueba (+ usuarios pm_demo y auditor_demo)
python app.py                # http://localhost:5000/admin/login
```

Vista previa sin tocar `data/`: `python .claude/preview_sgep.py` (puerto 5059, `/_dev_login?rol=pm|auditor|admin`).

## Tests

```bash
python -m pytest tests/test_sgep.py -q
```

Cubren NPS, avance esperado, cumplimiento, participación, semáforo, filtros del dashboard, validación del editor y de respuestas (incluida la lógica condicional), permisos por rol (pm, auditor, operador, sin sesión, aislamiento de RAC), CSRF en formularios internos y en la encuesta pública, link único, anonimato, encuesta cerrada, invitaciones/recordatorios y exportaciones.

## Despliegue

No requiere pasos extra: el blueprint se registra en `app.py`. En producción hay que reconstruir la imagen (por `openpyxl` y `reportlab`) y configurar las variables SMTP si se quieren correos reales. **No correr `sgep_seed.py` en producción.**
