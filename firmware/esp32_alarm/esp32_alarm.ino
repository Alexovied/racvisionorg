#include <Arduino.h>

/*
 * Detector de Casco - ESP32 Hardware Unit
 * Proyecto: RAC Vision
 * 
 * Pines:
 * - LED Verde: GPIO 18 (Acceso Seguro - Se activa con 'G' del historial)
 * - LED Rojo:  GPIO 19 (Alerta Infracción - Se activa con 'R' del historial)
 * - Buzzer:    GPIO 23 (Alarma Sonora)
 * - Pulsador:  GPIO 21 (Silenciar Buzzer)
 */

const int PIN_LED_VERDE = 18;
const int PIN_LED_ROJO  = 19;
const int PIN_BUZZER    = 23;
const int PIN_BOTON     = 21;

enum EstadoSeguridad { SEGURO, ALERTA };
EstadoSeguridad estadoActual = SEGURO;

bool buzzerSilenciado = false;
int estadoReposoBoton = HIGH;

void aplicarActuadores() {
  if (estadoActual == ALERTA) {
    digitalWrite(PIN_LED_VERDE, LOW);
    digitalWrite(PIN_LED_ROJO, HIGH);
    digitalWrite(PIN_BUZZER, buzzerSilenciado ? LOW : HIGH);
  } else {
    digitalWrite(PIN_LED_VERDE, HIGH);
    digitalWrite(PIN_LED_ROJO, LOW);
    digitalWrite(PIN_BUZZER, LOW);
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_ROJO, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_BOTON, INPUT_PULLUP);

  // Autodiagnóstico inicial (250 ms)
  digitalWrite(PIN_LED_VERDE, HIGH);
  digitalWrite(PIN_LED_ROJO, HIGH);
  digitalWrite(PIN_BUZZER, HIGH);
  delay(250);

  // Calibrar estado de reposo del pulsador (funciona tanto a GND como a 3.3V)
  estadoReposoBoton = digitalRead(PIN_BOTON);

  // Estado seguro por defecto en espera
  estadoActual = SEGURO;
  buzzerSilenciado = false;
  aplicarActuadores();

  Serial.println("ESP32_SAFETY_READY");
}

void loop() {
  // 1. Comandos Serial desde Python / Historial
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'R') { // ALERTA: Sin Casco (registrado en historial)
      estadoActual = ALERTA;
      buzzerSilenciado = false; // Siempre encender buzzer
      aplicarActuadores();
      Serial.println("ESTADO:ALERTA_SIN_CASCO");
    } 
    else if (cmd == 'G') { // SEGURO: Con Casco (registrado en historial)
      estadoActual = SEGURO;
      buzzerSilenciado = false; // Apagar alerta y prender LED Verde
      aplicarActuadores();
      Serial.println("ESTADO:ACCESO_SEGURO");
    }
  }

  // 2. Pulsador para silenciar buzzer inmediatamente
  if (estadoActual == ALERTA && !buzzerSilenciado) {
    // Si se presiona el botón (cambia de su nivel de reposo)
    if (digitalRead(PIN_BOTON) != estadoReposoBoton) {
      buzzerSilenciado = true;
      digitalWrite(PIN_BUZZER, LOW); // Silenciar sonido al instante
      Serial.println("EVENTO:BUZZER_SILENCIADO_POR_PULSADOR");
    }
  }
}
