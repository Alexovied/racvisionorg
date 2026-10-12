# RAC Vision × Ekoparty

> **Tesis:** RAC Vision es un sistema de *seguridad laboral* (safety) con IA y un ESP32. Si alguien puede apagar la alarma, falsificar el registro o esquivar el modelo, el sistema no solo deja de proteger: además **da una falsa sensación de seguridad**. Por eso la seguridad informática (security) de un sistema de safety es parte de su función, no un extra.

Este documento explica por qué RAC Vision encaja en Ekoparty, qué parte de esa conexión ya está aplicada en el código, qué falta y cómo se mostraría en una demo. Lo marcado **(a verificar)** surge de leer el código y no se comprobó ejecutándolo.

---

## 1. ¿Existe la conexión con Ekoparty?

**Existía a medias.** Antes de este trabajo, el vínculo era solo de formación: la revisión de seguridad (`seguridad-rac` en el vault) recomendaba los cursos *Safe Code Essentials* y *Bug Bounty* de Ekoparty Hackademy. El sistema en sí no tenía nada que mostrar en una conferencia de seguridad: era un detector de cascos con endpoints abiertos.

**Ahora queda aplicada** de esta forma:

| Parte de la conexión | Estado | Justificación |
|---|---|---|
| Revisión de seguridad con hallazgos numerados (#1–#10) | ✅ Hecha (vault) | Es el punto de partida de cualquier charla de *attack & defense*: qué estaba mal y por qué importa. |
| Arreglos #1, #4 y #5 (credenciales por defecto, debugger, secretos en git) | ✅ Aplicados (paso 1) | Son los errores de configuración más comunes en IoT/OT. Muestran lo barato que es cerrarlos. |
| Arreglos #2, #3 y #6 (autenticación de API, validación de eventos, límite de tamaño) | ✅ Aplicados (paso 2) | Son el núcleo de la demo: **apagar una alarma física con un solo `POST`** era posible sin autenticarse. |
| Pruebas automáticas que demuestran cada defensa | ✅ `tests/test_seguridad.py` | Sirven como evidencia reproducible en vivo (`pytest -q`). |
| CSRF y rate limit (#7 y #8) | ⏳ Paso 3 | Completan la defensa del panel. |
| Hallazgos nuevos de *safety* (#11–#15, abajo) | 🆕 Documentados, sin arreglar | Son lo más original para Ekoparty: van más allá de la seguridad web clásica. |
| Este documento y el guion de demo | ✅ Este archivo | Ordena el material para presentarlo. |

---

## 2. Por qué encaja en Ekoparty

La convocatoria de Ekoparty Buenos Aires 2026 lista, entre otras, estas áreas temáticas. RAC Vision toca cuatro:

| Área de Ekoparty | Qué aporta RAC Vision |
|---|---|
| **AI** | Clasificador MobileNetV2 (transfer learning) más detección de caras Haar. Se puede atacar por evasión (no mostrar la cara) y, posiblemente, con objetos que se parecen a un casco **(a verificar)**. |
| **IoT / Smart Devices** | ESP32 con LEDs, buzzer y pulsador, controlado por USB serie sin autenticación. |
| **Industrial Control Systems / SCADA** | Es un sistema de safety para minería: falla segura vs. falla abierta, alarmas, auditoría. |
| **DevSecOps** (espacio propio) | Secretos fuera de git, configuración por variables de entorno y tests de seguridad automatizados. |

Lo que lo hace interesante no es un exploit sofisticado, sino la **cadena completa**: navegador → API Flask → modelo de IA → puerto serie → actuador físico, con fallas en cada eslabón.

---

## 3. Modelo de amenazas

**Activos:** la alarma física (LED rojo + buzzer), el registro de auditoría (`events.json`), las cuentas del panel y la privacidad de las personas filmadas.

**Actores:**
- **A1. Alguien en la red** que llega al puerto del servidor.
- **A2. Operador autenticado malintencionado**, o una sesión robada.
- **A3. Trabajador que quiere evitar la alarma** (no es un atacante informático, pero es el caso más real).
- **A4. Acceso físico** al servidor o al ESP32 (USB).

**Frontera de confianza clave:** hoy **la decisión "con casco / sin casco" se toma en el navegador** (ventana deslizante en `templates/index.html`), y el servidor confía en lo que el navegador le informa. Pedir login (paso 2) evita que cualquiera falsifique eventos, pero un operador autenticado todavía puede mandar `con_casco` a mano.

### Hallazgos de la revisión original (vault)

| # | Hallazgo | Actor | Estado |
|---|---|---|---|
| 1 | Admin por defecto `admin` / `admin123` | A1 | ✅ Paso 1 |
| 2 | `/api/alarm` sin autenticación: se apaga la alarma con un `POST` | A1 | ✅ Paso 2 |
| 3 | `POST /admin/api/events` sin autenticación: log poisoning | A1 | ✅ Paso 2 |
| 4 | `debug=True` con `host=0.0.0.0`: ejecución remota de código por el debugger de Werkzeug | A1 | ✅ Paso 1 |
| 5 | `data/` fuera de `.gitignore` | — | ✅ Paso 1 |
| 6 | `/predict` público y sin límite de tamaño: DoS por CPU | A1 | ✅ Paso 2 |
| 7 | Sin CSRF en el panel | A1/A2 | ⏳ Paso 3 |
| 8 | Login sin límite de intentos | A1 | ⏳ Paso 3 |
| 9 | JSON como base de datos (condiciones de carrera, disco efímero en Render) | — | Pendiente |
| 10 | Privacidad: hay que dejar escrito que no se guardan frames | — | Paso 5 (README) |

### Hallazgos nuevos (encontrados al preparar esta presentación)

| # | Severidad | Hallazgo | Dónde | Arreglo propuesto |
|---|---|---|---|---|
| 11 | **Alta (safety)** | **Falla abierta cuando no hay cara.** Si Haar no detecta un rostro, el estado `NO_FACE` no cambia la decisión: si lo último fue "ACCESO SEGURO", **queda en verde**. Alguien sin casco que mira hacia abajo o se tapa la cara no dispara la alarma. **(a verificar)** | `index.html` (`updateDashboard`) y `app.py` (`crop_head`) | Si hay movimiento o persona pero no cara durante N segundos, pasar a un estado "indeterminado" que no sea verde. Evaluar un detector de personas o cabezas en vez de caras. |
| 12 | **Alta (safety)** | **El ESP32 arranca en verde y no tiene watchdog.** El firmware inicia en `SEGURO` y mantiene el último estado para siempre. Si se cae el servidor, el navegador o el cable USB, la luz verde queda prendida. | `firmware/esp32_alarm/esp32_alarm.ino` | Arrancar en un estado neutro o de alerta. Sumar un *heartbeat*: si no llega un comando en X segundos, pasar a un estado de falla visible. |
| 13 | Media | **El comando `O` no existe en el firmware.** `hardware.py` y `/api/alarm` documentan `O` como "apagar todo", pero el `loop()` solo atiende `R` y `G`. `{"state": "off"}` responde `ok` y no hace nada. | firmware y `app.py` | Implementar `O` en el firmware o sacarlo de la API, y documentar el protocolo real. |
| 14 | Media | **La decisión de safety se toma en el cliente.** El navegador decide y el servidor obedece. Un operador autenticado (A2) puede falsificar `con_casco`. | `index.html` → `/admin/api/events` y `/api/alarm` | Mover la decisión (ventana y umbrales) al servidor, a partir de los resultados de `/predict`, y quitar `/api/alarm` del alcance del navegador. |
| 15 | Media (privacidad) | **Hay imágenes de debug en git:** `debug_crop.png` y `debug_cropped_color.jpg`. Por el nombre, parecen recortes de cabeza. No se abrieron para esta revisión. **(a verificar)** | raíz del repo | Si tienen caras, sacarlas del repo (y del historial si es necesario) y agregar `debug_*` a `.gitignore`. |

Los hallazgos #11 y #12 son el aporte más fuerte para Ekoparty: en un sistema de safety, **la pregunta no es solo "¿pueden entrar?", sino "¿qué pasa cuando algo falla?"**. Hoy la respuesta es "queda en verde".

---

## 4. Guion de demo (formato Live Demo, 15–20 min)

Todo se hace **contra una instancia propia y local**, sin datos reales de personas.

1. **Contexto (2 min).** Qué es RAC Vision: el diagrama del flujo navegador → Flask → MobileNetV2 → ESP32. El LED verde es lo que el trabajador ve.
2. **Ataque 1: apagar la alarma (3 min).** En la versión anterior a los arreglos (`git checkout <commit-anterior>`), con el LED rojo encendido, un solo `curl -X POST http://localhost:5000/api/alarm -H "Content-Type: application/json" -d "{\"state\":\"con_casco\"}"` lo pasa a verde **sin login**.
3. **Ataque 2: falsificar la auditoría (2 min).** Lo mismo contra `/admin/api/events`: el panel muestra eventos que nunca ocurrieron.
4. **Defensa (4 min).** En la versión actual, los mismos comandos devuelven `401`, y un evento con `type` inválido devuelve `400`. Se corre `pytest -q` en vivo para mostrar las pruebas.
5. **Ataque 3: evadir el modelo sin tocar la red (4 min).** Con el LED en verde, una persona sin casco baja la cabeza o se tapa la cara: queda en verde (#11). Desenchufar el USB también lo deja en verde (#12).
6. **Lecciones (3 min).** *Fail-safe* vs. *fail-open*, que la decisión de safety viva en el servidor, y el checklist de arreglos baratos (#1, #4 y #5 llevan unos 15 minutos).

> Para el paso 2 de la demo hace falta el hash del commit anterior a los arreglos. Se completa cuando estén hechos los commits.

---

## 5. Cómo y cuándo presentarlo (realista)

- **Ekoparty Buenos Aires 2026 es del 7 al 9 de octubre de 2026** (empieza hoy). Su convocatoria a charlas cerró el 14 de agosto y la del espacio DevSecOps, el 23 de agosto. **Este año no hay forma de entrar como speaker por CFP.** Sí se puede llevar la demo para mostrarla informalmente, preguntar en los villages si aceptan demos y hablar con la gente de Hackademy.
- **Próximas ediciones (estimado, según el calendario de 2026):** la convocatoria de Ekoparty Miami 2026 abrió el 6 de noviembre de 2025 y cerró el 20 de marzo de 2026. La de Buenos Aires 2026 fue del 3 de junio al 14 de agosto. Si se repite, la de **Miami 2027 abriría alrededor de noviembre de 2026** y la de **Buenos Aires 2027, alrededor de junio de 2027**. Hay que confirmarlo cuando se publiquen.
- **Formatos que encajan:** *Talk* de hasta 30 minutos o *Live Demo* de 15–20 minutos (según el espacio DevSecOps 2026). La convocatoria principal prioriza las propuestas que **adjuntan documentación técnica y slides**. Este archivo, el README y los tests son esa base.

> [!important] Política de IA de Ekoparty
> La convocatoria de Ekoparty Miami 2026 aclara que **las propuestas generadas total o parcialmente con herramientas de IA se descalifican automáticamente**. Por eso este documento **no incluye un abstract ni un título listo para enviar**. Es material técnico de apoyo. La propuesta (título, resumen y bio) tiene que escribirla el autor con sus palabras.

### Ejes para que el autor escriba su propuesta
- El problema: un sistema de safety con IA que da falsa seguridad.
- Tres capas atacadas: la API web, el modelo de IA (evasión) y el hardware (falla abierta).
- Evidencia reproducible: el antes y el después en git, más las pruebas automatizadas.
- La lección general, aplicable a cualquier sistema de IoT u OT: falla segura, decisión en el servidor y configuración segura por defecto.

---

## 6. Ética y alcance

- Todas las pruebas se hacen sobre el sistema propio, en local. No se ataca ningún sistema de terceros.
- No se guardan frames: `/predict` procesa la imagen en memoria y solo se registran metadatos (tipo, confianza, cámara y hora). Ver la política de privacidad en el README (paso 5).
- Para la demo en vivo se usan voluntarios que dan su consentimiento o imágenes de prueba, nunca grabaciones de trabajadores reales.

---

## Fuentes
- [Ekoparty Security Conference 2026 (Buenos Aires): Call for Speakers](https://sessionize.com/ekoparty-security-conference-2026-buenos-aires)
- [Ekoparty 2026: DevSecOps Space, Call for Speakers](https://sessionize.com/ekoparty-2026-devsecops-space)
- [Ekoparty Security Conference 2026 Miami: Call for Speakers](https://sessionize.com/ekoparty-security-conference-2026-miami/)
- [ekoparty.org](https://ekoparty.org/)
- Revisión interna: `seguridad-rac`, `arquitectura-rac` y `codigo-racvision` (vault de Obsidian del autor).
