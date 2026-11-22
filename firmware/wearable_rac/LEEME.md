# Wearable RAC: pulso, caídas, botón de pánico y hombre muerto

Pulsera o arnés que cada trabajador lleva puesto. Manda sus datos a RAC Vision cada 5 segundos y
el panel avisa (aviso rojo en todas las páginas, Wearables, Auditoría, Telegram, n8n y la sirena de
la caja RAC) cuando:

| Alerta | Cuándo |
|---|---|
| Botón de pánico | el trabajador mantiene apretado el botón 2 s |
| Hombre muerto | queda quieto 30 s, el equipo vibra y pita 15 s, y nadie aprieta el botón |
| Posible desmayo | quieto 30 s después de una caída, o quieto con el pulso bajo / sin pulso |
| Ritmo cardíaco | menos de 40 o más de 160 lpm, sin pulso con el sensor apoyado, u oxígeno < 88 %, 20 s seguidos |
| Sin señal | el equipo no manda nada hace más de 90 s (batería, golpe, sin cobertura) |

Si la cámara de la zona del trabajador vio a alguien en el piso en los mismos 2 minutos, la alerta
llega **confirmada por la cámara** (y al revés: una caída vista por cámara llega confirmada por el wearable).

Cuando el panel recibe una alerta, el equipo vibra dos veces cortas: el trabajador sabe que ya se enteraron.

## Probar sin hardware

1. Levantar el panel y entrar como admin. **Wearables → Registrar un equipo** y copiar el token.
2. `python simulador_wearable.py --token rac_XXXX --escenario desmayo`
   Escenarios: `normal`, `sos`, `hombre_muerto`, `desmayo`, `taquicardia`, `sin_pulso`, `sin_senal`.
3. En el panel aparece el aviso rojo con «Atendida: fue real» / «Falsa alarma».

## Probar el firmware en Wokwi

1. Abrir https://wokwi.com/projects/new/esp32, pegar `wearable_rac.ino` y `diagram.json` (y `libraries.txt`).
2. Con `SIMULAR_PULSO 1` el potenciómetro hace de pulso (a la izquierda = 0 = sin pulso).
3. Tocar ▶. En el monitor serie se ve `WEARABLE_RAC_LISTO` y cada envío.
4. Mantener el botón rojo 2 s → `SOS`. Dejar quieto el MPU6050 30 s → `HM_AVISO` (pita y se prende el LED
   «Vibrador»); si no se aprieta el botón en 15 s → `HOMBRE_MUERTO`.
5. Para que llegue al panel, `SERVIDOR` tiene que ser una dirección pública (Wokwi no ve tu `localhost`):
   un túnel tipo ngrok a tu PC, o el servidor de pruebas. No apuntarlo al de producción para probar.

> El firmware todavía no se compiló en un ESP32 real ni en Wokwi: es la primera versión de referencia.

## Armado (equipo real)

| Pieza | Pin del ESP32 |
|---|---|
| MAX30102 (pulso y oxígeno) | SDA 21, SCL 22, 3V3, GND |
| MPU6050 (acelerómetro) | SDA 21, SCL 22, 3V3, GND (mismo bus I2C) |
| Pulsador | GPIO 4 a GND |
| Buzzer | GPIO 23 |
| Motor vibrador | GPIO 19 por un transistor (2N2222 + diodo 1N4148) |
| Batería LiPo 3,7 V | por cargador TP4056 con protección; divisor 100k/100k a GPIO 35 para medir la carga |

Poner `SIMULAR_PULSO` en `0`, cargar `WIFI_SSID`, `WIFI_PASS`, `SERVIDOR` y `TOKEN`, e instalar la
biblioteca de `libraries.txt` desde el Administrador de bibliotecas del IDE de Arduino.

### Lista de materiales (precios aproximados por unidad, nov. 2026, compra minorista)

| Pieza | USD aprox. |
|---|---|
| ESP32 DevKit (o ESP32-C3 SuperMini, más chico) | 4 a 8 |
| MAX30102 (pulso + SpO₂) | 3 a 5 |
| MPU6050 (acelerómetro y giroscopio) | 2 a 4 |
| Batería LiPo 500 a 1000 mAh + TP4056 | 5 a 8 |
| Motor vibrador, buzzer, pulsador, transistor | 2 a 3 |
| Caja impresa en 3D + correa | 3 a 6 |
| **Total por equipo (prototipo)** | **~20 a 35** |

Opcionales: módulo LoRa SX1276/RFM95 (US$ 6 a 10 por equipo) más un gateway LoRa (US$ 70 a 150) para
zonas sin Wi-Fi; banda de pecho Polar H10 (~US$ 90) si el pulso de muñeca no es confiable.

## Cosas a tener en cuenta antes de usarlo en una mina

- **El pulso en la muñeca con movimiento es poco confiable** con el MAX30102. Para pruebas sirve; para
  uso real conviene sensor en el pecho (banda) o en el arnés, apoyado firme contra la piel.
- **Bajo tierra casi nunca hay Wi-Fi.** La alternativa es LoRa (alcance de cientos de metros en galería,
  más con repetidores) o una red mesh. El protocolo es el mismo: un gateway LoRa recibe los datos y los
  reenvía por HTTP a `/admin/api/dispositivos/telemetria` con el token de cada equipo.
- **Atmósferas explosivas:** en minería subterránea los equipos electrónicos tienen que ser
  intrínsecamente seguros (certificación ATEX / IECEx). Un prototipo casero no lo es: sirve para
  validar la idea; para producción hay que usar hardware certificado o pasar el diseño por certificación.
- No reemplaza la supervisión ni los procedimientos de rescate: es un aviso más.
