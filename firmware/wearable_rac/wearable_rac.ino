/*
 * Wearable RAC: pulsera o arnés para el trabajador (ESP32)
 *
 * Mide y manda a RAC Vision cada 5 segundos por Wi-Fi:
 *   - Pulso (MAX30102 apoyado en la piel) y si el sensor está tocando la piel
 *   - Actividad y caídas (acelerómetro MPU6050): caída libre seguida de un golpe = caída
 *   - Botón de pánico: mantener apretado 2 s
 *   - Hombre muerto: si queda quieto HM_QUIETO_MS, vibra y pita HM_AVISO_MS. Si el
 *     trabajador no aprieta el botón en ese tiempo, avisa. Apretarlo corta el aviso.
 * Las alertas urgentes (SOS, caída, hombre muerto) salen al instante, sin esperar los 5 s.
 * Cuando el panel confirma que recibió una alerta, el equipo vibra dos veces cortas: el
 * trabajador sabe que alguien ya se enteró.
 *
 * Las reglas de pulso (bradicardia, taquicardia, sin pulso, desmayo) las decide el servidor
 * (wearables.py): acá solo se mide. Así se ajustan desde el panel sin reprogramar equipos.
 *
 * Bibliotecas: "SparkFun MAX3010x Pulse and Proximity Sensor Library" (Administrador de
 * bibliotecas del IDE de Arduino). El MPU6050 se lee directo por I2C, sin biblioteca.
 *
 * Simulador Wokwi (sin MAX30102): dejar SIMULAR_PULSO en 1 y el pulso sale del potenciómetro.
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>

#define SIMULAR_PULSO 1   // 1 = Wokwi / sin sensor de pulso: el potenciómetro hace de pulso. 0 = MAX30102 real

#if !SIMULAR_PULSO
#include "MAX30105.h"
#include "heartRate.h"
MAX30105 sensorPulso;
#endif

// ── Configurar antes de grabar ────────────────────────────────────────────────
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";
// Dirección del panel + ruta de telemetría. Ej: "https://tu-servidor/admin/api/dispositivos/telemetria"
const char* SERVIDOR = "http://192.168.0.10:5000/admin/api/dispositivos/telemetria";
// Token del equipo: Panel → Wearables → Registrar un equipo (se muestra una sola vez)
const char* TOKEN = "rac_PEGAR_TOKEN_AQUI";

// ── Pines ─────────────────────────────────────────────────────────────────────
const int PIN_BOTON = 4;        // a GND, con pull-up interno
const int PIN_BUZZER = 23;
const int PIN_VIBRADOR = 19;    // motor vibrador por un transistor (en Wokwi, un LED)
const int PIN_POT_PULSO = 34;   // solo con SIMULAR_PULSO
const int PIN_BATERIA = 35;     // divisor 100k/100k desde la batería (mide la mitad)
const uint8_t MPU = 0x68;       // I2C: SDA 21, SCL 22 (compartido con el MAX30102)

// ── Tiempos y umbrales del equipo ───────────────────────────────────────────
const unsigned long ENVIO_MS = 5000;
const unsigned long SOS_MS = 2000;           // mantener apretado para pedir ayuda
const unsigned long HM_QUIETO_MS = 30000;    // quieto este tiempo → empieza a avisar
const unsigned long HM_AVISO_MS = 15000;     // tiempo para apretar el botón antes de alertar
const float QUIETO_G = 0.05;                 // actividad por debajo de esto = quieto (igual que el servidor)
const float CAIDA_LIBRE_G = 0.4, GOLPE_G = 2.5;

// ── Estado ──────────────────────────────────────────────────────────────────
float actividad = 0.5;            // promedio móvil de |aceleración - 1 g|
unsigned long quietoDesde = 0, avisoDesde = 0, caidaLibreEn = 0, ultimoEnvio = 0;
unsigned long botonDesde = 0;
bool caidaPendiente = false, sosPendiente = false, hombreMuerto = false, hmEnviado = false, enAviso = false;
float bpm = 0;
bool contacto = false;

#if !SIMULAR_PULSO
const byte LATIDOS = 4;
byte latidos[LATIDOS];
byte latidoN = 0;
unsigned long ultimoLatido = 0;
#endif

void mpuEscribir(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

float leerAceleracionG() {
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return 1.0;
  Wire.requestFrom(MPU, (uint8_t)6);
  if (Wire.available() < 6) return 1.0;
  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  int16_t az = (Wire.read() << 8) | Wire.read();
  const float LSB_G = 4096.0;   // rango ±8 g
  return sqrtf(sq(ax / LSB_G) + sq(ay / LSB_G) + sq(az / LSB_G));
}

int bateriaPorcentaje() {
  float v = analogReadMilliVolts(PIN_BATERIA) * 2 / 1000.0;   // divisor a la mitad
  if (v < 1.0) return -1;                                      // no hay divisor conectado
  return constrain((int)((v - 3.3) / (4.2 - 3.3) * 100), 0, 100);
}

void vibrar(int veces, int ms) {
  for (int i = 0; i < veces; i++) {
    digitalWrite(PIN_VIBRADOR, HIGH);
    delay(ms);
    digitalWrite(PIN_VIBRADOR, LOW);
    if (i < veces - 1) delay(ms);
  }
}

void conectarWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
  Serial.println(WiFi.status() == WL_CONNECTED ? "WIFI_OK " + WiFi.localIP().toString() : String("WIFI_SIN_CONEXION"));
}

void medirPulso() {
#if SIMULAR_PULSO
  bpm = map(analogRead(PIN_POT_PULSO), 0, 4095, 0, 200);   // potenciómetro: 0 = sin pulso
  contacto = true;
#else
  long ir = sensorPulso.getIR();
  contacto = ir > 50000;
  if (!contacto) { bpm = 0; latidoN = 0; return; }
  if (checkForBeat(ir)) {
    unsigned long ahora = millis();
    float instante = 60000.0 / (ahora - ultimoLatido);
    ultimoLatido = ahora;
    if (instante > 20 && instante < 250) {
      latidos[latidoN++ % LATIDOS] = (byte)instante;
      byte n = min(latidoN, LATIDOS);
      int suma = 0;
      for (byte i = 0; i < n; i++) suma += latidos[i];
      bpm = suma / (float)n;
    }
  }
  if (millis() - ultimoLatido > 5000) bpm = 0;   // apoyado y sin latidos hace 5 s
#endif
}

void medirMovimiento(unsigned long ahora) {
  float g = leerAceleracionG();
  actividad = actividad * 0.95 + fabsf(g - 1.0) * 0.05;
  if (g < CAIDA_LIBRE_G) caidaLibreEn = ahora;
  if (g > GOLPE_G && caidaLibreEn && ahora - caidaLibreEn < 1000) {
    caidaPendiente = true;
    caidaLibreEn = 0;
    Serial.println("CAIDA");
  }
}

void revisarBoton(unsigned long ahora) {
  bool apretado = digitalRead(PIN_BOTON) == LOW;
  if (!apretado) { botonDesde = 0; return; }
  if (!botonDesde) {
    botonDesde = ahora;
    // Toque corto: "estoy bien". Corta el aviso de hombre muerto
    if (enAviso || hombreMuerto) {
      enAviso = false;
      hombreMuerto = false;
      hmEnviado = false;
      quietoDesde = ahora;
      noTone(PIN_BUZZER);
      digitalWrite(PIN_VIBRADOR, LOW);
      Serial.println("HM_CANCELADO");
    }
  } else if (ahora - botonDesde >= SOS_MS && !sosPendiente) {
    sosPendiente = true;
    Serial.println("SOS");
    vibrar(1, 400);
  }
}

void revisarHombreMuerto(unsigned long ahora) {
  if (actividad >= QUIETO_G) {
    quietoDesde = ahora;
    if (enAviso) { enAviso = false; noTone(PIN_BUZZER); digitalWrite(PIN_VIBRADOR, LOW); }
    return;
  }
  if (!enAviso && !hombreMuerto && ahora - quietoDesde >= HM_QUIETO_MS) {
    enAviso = true;
    avisoDesde = ahora;
    Serial.println("HM_AVISO");
  }
  if (enAviso) {
    bool fase = (ahora / 500) % 2;   // vibra y pita intermitente
    digitalWrite(PIN_VIBRADOR, fase);
    if (fase) tone(PIN_BUZZER, 2000); else noTone(PIN_BUZZER);
    if (ahora - avisoDesde >= HM_AVISO_MS) {
      enAviso = false;
      hombreMuerto = true;     // queda hasta que alguien apriete el botón
      tone(PIN_BUZZER, 2500);  // pitido fijo: que lo encuentren
      Serial.println("HOMBRE_MUERTO");
    }
  }
}

bool enviar() {
  conectarWifi();
  if (WiFi.status() != WL_CONNECTED) return false;
  int bat = bateriaPorcentaje();
  String json = "{\"bpm\":" + String((int)bpm) +
                ",\"contacto\":" + (contacto ? "true" : "false") +
                ",\"actividad\":" + String(actividad, 3) +
                ",\"caida\":" + (caidaPendiente ? "true" : "false") +
                ",\"sos\":" + (sosPendiente ? "true" : "false") +
                ",\"hombre_muerto\":" + (hombreMuerto ? "true" : "false") +
                (bat >= 0 ? ",\"bateria\":" + String(bat) : String("")) + "}";
  HTTPClient http;
  http.begin(SERVIDOR);   // para https con certificado propio: WiFiClientSecure + setCACert
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", String("Bearer ") + TOKEN);
  http.setTimeout(4000);
  int codigo = http.POST(json);
  String respuesta = codigo > 0 ? http.getString() : "";
  http.end();
  Serial.printf("ENVIO %d %s\n", codigo, json.c_str());
  if (codigo != 200) return false;
  // Lo que ya llegó al panel no se vuelve a mandar como nuevo
  caidaPendiente = false;
  sosPendiente = false;
  hmEnviado = hombreMuerto;
  if (respuesta.indexOf("\"aviso_recibido\":true") >= 0) vibrar(2, 150);   // "ya se enteraron"
  return true;
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_BOTON, INPUT_PULLUP);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_VIBRADOR, OUTPUT);
  Wire.begin(21, 22);
  mpuEscribir(0x6B, 0x00);   // despertar el MPU6050
  mpuEscribir(0x1C, 0x10);   // acelerómetro ±8 g (un golpe fuerte pasa los 2 g)
#if !SIMULAR_PULSO
  if (!sensorPulso.begin(Wire, I2C_SPEED_FAST)) Serial.println("MAX30102_NO_ENCONTRADO");
  sensorPulso.setup();
  sensorPulso.setPulseAmplitudeRed(0x0A);
  sensorPulso.setPulseAmplitudeGreen(0);
#endif
  quietoDesde = millis();
  conectarWifi();
  Serial.println("WEARABLE_RAC_LISTO");
}

void loop() {
  unsigned long ahora = millis();
  medirPulso();
  medirMovimiento(ahora);
  revisarBoton(ahora);
  revisarHombreMuerto(ahora);

  // Lo urgente sale ya (y si falla, se reintenta cada segundo, no en cada vuelta)
  bool urgente = (caidaPendiente || sosPendiente || (hombreMuerto && !hmEnviado)) && ahora - ultimoEnvio > 1000;
  if (urgente || ahora - ultimoEnvio >= ENVIO_MS) {
    enviar();               // si falla, las banderas quedan y se reintenta en la próxima vuelta
    ultimoEnvio = ahora;
  }
  delay(20);   // ~50 lecturas por segundo: alcanza para el acelerómetro y el pulso
}
