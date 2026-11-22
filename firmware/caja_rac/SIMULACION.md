# Caja RAC: alarma de obra (diseño y simulación)

Versión mejorada de `esp32_alarm`, pensada para instalar en una obra: distingue qué falta, se ve y se escucha de lejos, recibe las alertas por Wi-Fi y registra cuando un supervisor las reconoce.

| Componente | Pin del ESP32 | Para qué |
|---|---|---|
| Torre de señalización verde / amarilla / roja | GPIO 25 / 26 / 27 | Todos con EPP / sin chaleco / sin casco |
| Relé 1 → baliza giratoria de 12 V | GPIO 32 | Intermitente con cualquier alerta |
| Relé 2 → sirena de 12 V | GPIO 33 | Solo sin casco, hasta que la reconozcan |
| Buzzer local | GPIO 23 | Sirena de dos tonos en la caja |
| Pulsador "Reconocer" | GPIO 4 | El supervisor confirma que vio la alerta y se apaga la sirena |
| Pantalla LCD 16x2 I2C | SDA 21, SCL 22 | Estado y dirección IP |

En la simulación, la baliza y la sirena de 12 V se muestran como LEDs conectados al contacto de cada relé. En la obra, cada relé corta los 12 V de una fuente aparte, no los 5 V del ESP32.

## Comandos
Un carácter por el puerto serie (115200 baudios) o por Wi-Fi con `GET http://<ip>/estado?c=R`. `R` y `G` son los mismos de `esp32_alarm`, así que RAC ya los manda sin cambios.

| Comando | Estado |
|---|---|
| `G` | Todos con EPP: verde |
| `V` | Sin chaleco: amarillo y baliza |
| `R` | Sin casco: rojo, baliza y sirena |
| `O` | Apagado |

`GET /reconocer` hace lo mismo que el pulsador. Hoy RAC (`admin.py`) manda solo `R` y `G`: para el amarillo hay que sumar que mande `V` cuando falte solo el chaleco.

## Cómo simularla en Wokwi
1. Abrir https://wokwi.com/projects/new/esp32.
2. Pegar `caja_rac.ino` en `sketch.ino` y `diagram.json` en `diagram.json`.
3. En **Library Manager**, agregar **LiquidCrystal I2C** (o crear el archivo `libraries.txt` con el de esta carpeta).
4. Tocar ▶ y escribir `R`, `V`, `G` u `O` en el monitor serie. El pulsador azul reconoce la alerta.

Probado el 20/10/2026 en Wokwi: se conecta al Wi-Fi y muestra la IP en la pantalla. `R` prende la torre roja, la baliza intermitente y la sirena; el pulsador apaga la sirena ("SIN CASCO OK"); `V` deja la torre amarilla y la baliza sin sirena; `G` vuelve al verde. La simulación se pausa si la pestaña o el panel del navegador quedan ocultos.
