# Simulación de la alarma en Wokwi

`diagram.json` es el circuito de la alarma (ESP32 DevKit C, LED verde en GPIO 18, LED rojo en GPIO 19, buzzer en GPIO 23 y pulsador en GPIO 21, con resistencias de 220 Ω) para el simulador [Wokwi](https://wokwi.com).

1. Abrir https://wokwi.com/projects/new/esp32 (no hace falta cuenta para simular).
2. Reemplazar `sketch.ino` por el contenido de `esp32_alarm.ino` y `diagram.json` por el de este archivo.
3. Tocar ▶. La primera compilación puede tardar un par de minutos por la fila del servidor gratuito.
4. En el monitor serie, escribir `R` y Enter: se prende el LED rojo y suena el buzzer, como cuando RAC detecta a alguien sin casco. El pulsador amarillo silencia el buzzer y `G` vuelve al LED verde.

Probado el 20/10/2026: arranca con `ESP32_SAFETY_READY` y responde a `R`, al pulsador y a `G`. Para guardar el proyecto y compartir un link hay que iniciar sesión en Wokwi.
