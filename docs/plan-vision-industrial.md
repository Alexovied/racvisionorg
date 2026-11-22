# Plan: de prototipo a producto industrial

Lo que sigue del análisis externo de RAC Vision que no entra en un cambio de código chico: necesita
reentrenar, hardware o decisiones de producto. Ordenado por impacto.

## Ya hecho (PR #5)

- Salud de cámaras: sin señal más de 60 s, lente tapada o desenfocada más de 30 s, CPU / RAM / temperatura.
- Cámaras IP RTSP con timeout de apertura y lectura y reconexión con espera creciente.
- Zonas exentas de EPP (oficina, pasillo seguro) y horario de exigencia por cámara.
- Alertas por Telegram con foto, MQTT (QoS 1) para SCADA / PLC, webhook n8n.
- Modo anónimo (sin nombres, caras pixeladas) y embeddings cifrados con `RAC_CLAVE_BIOMETRIA`.
- Mapa de riesgo, pronóstico, indicadores por turno con CSV para ISO 45001 y reporte diario.
- Demo: la página del detector ya era pública (modo demostración, sin guardar nada). Se sumó probar con un
  video o una foto para quien no tiene webcam.

## 1. Modelo unificado + más EPP (reentrenar)

**Notebook listo:** `entrenamiento/detector_epp_unificado_colab.ipynb`.
Un YOLOv8n con persona, casco / sin casco, chaleco, antiparras / sin antiparras y guantes / sin guantes,
con aumentos de mina (polvo, niebla, linternas, luz de sodio, poca luz, movimiento).

- Objetivo: una pasada de unos 50 ms en lugar de dos de unos 48 ms + 47 ms, y cajas del mismo cuadro.
- Criterio para pasarlo a producción: en las fotos de prueba propias, igual o mejor que los dos modelos
  actuales en casco y chaleco, y sin el error de pelo oscuro = casco.
- Arnés y protección auditiva no están en ningún dataset abierto con licencia usable: hace falta etiquetar
  fotos propias (300 a 500 por clase como mínimo).

## 2. Integrar el modelo unificado (código, después del paso 1)

- `helmet_detector.py`: leer las clases del ONNX (metadata `names`) y devolver cabezas y personas de una pasada.
- `epp.py`: asignar cada chaleco / guante a **una** persona (la de mayor solapamiento), no a la caja más grande
  que lo contiene; hoy el chaleco de un vecino puede contar para otro en una fila de gente.
- Dejar los dos modelos actuales como respaldo si falta `detector_epp.onnx`.

## 3. Seguimiento de personas (ByteTrack)

Hoy cada cuadro se juzga solo y la ventana de votos es por cámara. Con un id por persona:
- la alerta es de esa persona ("el de la izquierda lleva 5 s sin casco"), no de la escena;
- no se mezclan infracciones entre dos trabajadores superpuestos;
- el nombre reconocido se arrastra aunque la cara deje de verse.

Propuesta: un ByteTrack liviano (asociación por IoU + filtro de Kalman, sin dependencias nuevas, unas 150
líneas) en `motor.py`, alimentado con las cajas de persona; las cabezas se asignan a la persona que las contiene.

## 4. Edge offline con sincronización

Para minas sin conexión estable:
- Eventos en SQLite (reemplaza `events.json`; además resuelve el límite de un solo worker de gunicorn).
- Cola de salida con id del evento como clave: un worker reintenta mandar eventos y fotos al panel central
  por HTTPS cuando hay red; el panel central ignora ids repetidos.
- El nodo sigue alertando local (ESP32, MQTT a la red de la planta) sin internet.

## 5. Cámaras y hardware

- ONVIF: descubrir cámaras de la red (WS-Discovery) desde el panel en lugar de escribir la URL RTSP.
- Decodificación por hardware Intel Quick Sync (`cv2.CAP_PROP_HW_ACCELERATION`) para muchas cámaras por nodo.
- Molinetes y barreras: el firmware de la caja RAC ya tiene relés; falta un modo "punto de control" que
  habilite el relé solo con casco y chaleco confirmados (puede ir por MQTT o por el ESP32).

## 6. Producto y web

- WhatsApp Business: necesita cuenta de Meta verificada y plantillas aprobadas; Telegram ya cubre el caso.
- Landing: el análisis sugiere destacar casos donde el EPP sí cambia el resultado (atropellos por falta de
  alta visibilidad, golpes por caída de herramientas) además de los casos de Osinergmin. Es una decisión de
  mensaje para Alex.
- Validación: pasar de 63 fotos de casco y 85 de chaleco a 500+ con casos difíciles (gorras, capuchas,
  cascos abollados, mochilas que tapan el chaleco, cámaras cenitales).
