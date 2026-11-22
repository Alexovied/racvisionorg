# SGEP-Dashboard — Manual de usuario

Sistema de Gestión de Encuestas para la Evaluación de Proyectos de 7mo (Expotécnica).
Está dentro del panel de RAC Vision: se entra por `/admin/login` y, según el rol, se ve el menú **SGEP**.

## Roles

| Rol | Puede |
|-----|-------|
| **Admin** | Todo: SGEP completo, usuarios y roles, y el panel de RAC Vision. |
| **PM** | Cargar proyectos y responsables, crear y editar encuestas, invitar, recordar, cerrar encuestas, ver el dashboard y exportar. |
| **Auditor** | Solo lectura: dashboard, listados, correos enviados y exportaciones a PDF y Excel. |

PM y Auditor no ven ni usan nada de RAC Vision (cámaras, monitoreo, detector). Los usuarios se crean en **Usuarios** (solo Admin), eligiendo el rol PM o Auditor.

## 1. Proyectos y responsables

Menú **Proyectos**.

1. Cargá primero los **responsables** (nombre y email) en la parte de abajo.
2. Arriba, completá el formulario **Nuevo proyecto**: nombre, curso (p. ej. «7mo 1ra»), cliente, responsable, fecha de inicio y de fin.
3. Para editar, tocá el lápiz de la fila: el formulario se completa con los datos y al guardar se actualiza.
4. Para borrar, tocá la papelera. Un proyecto con encuestas no se puede borrar (primero borrá sus encuestas). Un responsable con proyectos tampoco.

Las fechas sirven para calcular el **avance esperado**: si pasó la mitad del plazo, se espera un 50 %.

## 2. Constructor de encuestas

Menú **Encuestas** → **Nueva encuesta**: elegí el proyecto y, si querés, una **plantilla** (Satisfacción del cliente, Avance percibido, Lecciones aprendidas o las que guardes vos). Se abre el editor.

En el editor:

- **Arrastrá** un tipo de pregunta desde «Tipos de pregunta» a la lista, o hacé **clic** para sumarlo al final.
- **Reordená** arrastrando las tarjetas o con los botones ↑ ↓ (sirven con teclado).
- Tipos: **NPS (0–10)**, **% de avance (0–100)**, **Escala 1–5**, **Sí/No**, **Opción múltiple** (opciones separadas por coma) y **Texto libre** (opcional para quien responde).
- **Lógica condicional**: en «Mostrar solo si…» elegí una pregunta anterior y escribí la respuesta que la activa (p. ej. `si`). La pregunta solo aparece si se cumple.
- **Anónima**: si la marcás, las respuestas no guardan el nombre de quien responde (sí se registra que respondió, para medir la participación).
- **Guardar también como plantilla**: guarda una copia reutilizable.

Las preguntas NPS alimentan el KPI **NPS** y las de % de avance alimentan **avance percibido** y **cumplimiento**.

## 3. Distribución y recolección

Al abrir una encuesta (nombre en la lista, o el ícono de compartir) se ve:

- **Link general y QR**: para proyectar o imprimir en el stand. Quien entra por ahí no queda identificado.
- **Invitar por email**: una persona por línea, `Nombre <email>` o solo el email. Cada una recibe un **link único** que admite **una sola respuesta**. Los emails inválidos o repetidos se saltean y se avisa.
- **Recordar a pendientes**: manda un recordatorio solo a quien no respondió y cuenta cuántos recordatorios recibió cada uno.
- **Cerrar / reabrir encuesta**: cerrada, ya no acepta respuestas.
- **Control de respuestas**: tabla con cada invitación, cuándo se mandó, recordatorios y si respondió.

Menú **Correos**: registro de todos los correos. Sin servidor SMTP configurado (modo demo) los correos no salen y quedan como «registrado»; con SMTP figuran como «enviado» o «error».

## 4. Dashboard

Menú **SGEP Dashboard**. Se actualiza solo cada 10 segundos.

- **Filtros**: proyecto, cliente, responsable, tipo de encuesta y rango de fechas. Todo el tablero (y las exportaciones) respeta los filtros.
- **KPIs**:
  - **NPS** = % promotores (9–10) − % detractores (0–6), de −100 a 100.
  - **Avance percibido** = la última respuesta de % de avance de cada proyecto, promediada.
  - **Cumplimiento** = avance percibido sobre el avance esperado por calendario el día de cada respuesta (tope 100 %).
  - **Participación** = invitaciones respondidas / invitaciones enviadas.
  - **Proyectos en riesgo** = cantidad en rojo en el semáforo.
- **Semáforo**: 🔴 rojo si NPS < 0 o cumplimiento < 60 %; 🟡 amarillo si NPS < 30 o cumplimiento < 85 %; 🟢 verde en otro caso; gris sin datos.
- **Gráficos**: barras (NPS por proyecto), torta (promotores/pasivos/detractores), líneas (NPS y avance por semana), mapa de calor (avance percibido por proyecto y semana) y últimos comentarios / lecciones aprendidas.

## 5. Reportes

En el dashboard, **Excel** descarga un `.xlsx` con tres hojas (KPIs, Proyectos, Respuestas en formato largo para tablas dinámicas) y **PDF** un reporte apaisado con los KPIs y el semáforo. Ambos usan los filtros activos.

## Responder una encuesta (persona invitada)

Abrís el link del email o escaneás el QR, contestás (las preguntas con * son obligatorias) y tocás **Enviar respuesta**. Si te equivocás en un valor, la página lo marca y conserva lo que ya escribiste. La página tiene el mismo panel de accesibilidad que el resto del sitio.

## Datos de prueba

Para la demo: `python sgep_seed.py` (o `--force` para reemplazar). Crea 6 proyectos, 18 encuestas, ~10 semanas de respuestas y los usuarios `pm_demo` / `pm-demo-2026` y `auditor_demo` / `auditor-demo-2026`. **Solo para uso local.**
