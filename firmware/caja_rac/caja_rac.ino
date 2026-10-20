/*
 * Caja RAC: alarma de obra para RAC Vision (ESP32)
 *
 * Recibe el estado del EPP por el puerto serie (como esp32_alarm) o por Wi-Fi,
 * desde la PC que corre RAC, y lo muestra con:
 * - Torre de señalización: verde = todos con EPP, amarillo = sin chaleco, rojo = sin casco
 * - Relé 1: baliza giratoria de 12 V (sin casco o sin chaleco)
 * - Relé 2: sirena de 12 V (solo sin casco, hasta que alguien la reconozca)
 * - Buzzer local y pantalla LCD 16x2 con el estado y la conexión
 * - Pulsador "Reconocer": el supervisor confirma que vio la alerta y se apaga la sirena
 *
 * Comandos (un carácter, compatibles con esp32_alarm): G = todo OK, V = sin chaleco,
 * R = sin casco, O = apagar. Por Wi-Fi: GET http://<ip>/estado?c=R
 */
#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Red de la obra. En el simulador Wokwi la red es "Wokwi-GUEST", sin contraseña.
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";

const int PIN_VERDE = 25, PIN_AMARILLO = 26, PIN_ROJO = 27;
const int PIN_RELE_BALIZA = 32, PIN_RELE_SIRENA = 33;
const int PIN_BUZZER = 23, PIN_BOTON = 4;   // I2C de la pantalla: SDA 21, SCL 22

enum Estado { APAGADO, TODO_OK, SIN_CHALECO, SIN_CASCO };  // "OK" ya lo define el ESP-IDF
Estado estado = APAGADO;
bool reconocida = false;
unsigned long desde = 0;   // cuándo empezó la alerta actual
int frecuencia = 0;        // tono que suena ahora en el buzzer (0 = callado)

LiquidCrystal_I2C lcd(0x27, 16, 2);
WebServer server(80);

void sonar(int f) {
  if (f == frecuencia) return;   // volver a llamar a tone() en cada vuelta lo entrecorta
  frecuencia = f;
  if (f) tone(PIN_BUZZER, f); else noTone(PIN_BUZZER);
}

const char* nombre(Estado e) {
  switch (e) {
    case TODO_OK: return "TODOS CON EPP";
    case SIN_CHALECO: return "SIN CHALECO";
    case SIN_CASCO: return "SIN CASCO";
    default: return "APAGADO";
  }
}

void mostrar() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(nombre(estado));
  if (reconocida) lcd.print(" OK");
  lcd.setCursor(0, 1);
  if (WiFi.status() == WL_CONNECTED) lcd.print(WiFi.localIP());
  else lcd.print("Sin WiFi: serie");
}

void cambiar(Estado nuevo) {
  if (nuevo != estado) reconocida = false;
  estado = nuevo;
  desde = millis();
  digitalWrite(PIN_VERDE, estado == TODO_OK);
  digitalWrite(PIN_AMARILLO, estado == SIN_CHALECO);
  digitalWrite(PIN_ROJO, estado == SIN_CASCO);
  if (estado != SIN_CASCO && estado != SIN_CHALECO) digitalWrite(PIN_RELE_BALIZA, LOW);
  digitalWrite(PIN_RELE_SIRENA, estado == SIN_CASCO && !reconocida);
  if (estado != SIN_CASCO) sonar(0);
  Serial.print("ESTADO:");
  Serial.println(nombre(estado));
  mostrar();
}

bool comando(char c) {
  switch (c) {
    case 'G': cambiar(TODO_OK); return true;
    case 'V': cambiar(SIN_CHALECO); return true;
    case 'R': cambiar(SIN_CASCO); return true;
    case 'O': cambiar(APAGADO); return true;
  }
  return false;
}

void reconocer() {
  if ((estado != SIN_CASCO && estado != SIN_CHALECO) || reconocida) return;
  reconocida = true;
  digitalWrite(PIN_RELE_SIRENA, LOW);
  sonar(0);
  Serial.println("EVENTO:ALERTA_RECONOCIDA");
  mostrar();
}

void setup() {
  Serial.begin(115200);
  int salidas[] = {PIN_VERDE, PIN_AMARILLO, PIN_ROJO, PIN_RELE_BALIZA, PIN_RELE_SIRENA};
  for (int p : salidas) pinMode(p, OUTPUT);
  pinMode(PIN_BOTON, INPUT_PULLUP);

  lcd.init();
  lcd.backlight();
  lcd.print("RAC Vision");
  // Autodiagnóstico: prende todo medio segundo
  for (int p : salidas) digitalWrite(p, HIGH);
  tone(PIN_BUZZER, 2000, 200);
  delay(500);
  noTone(PIN_BUZZER);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  server.on("/estado", []() {
    String c = server.arg("c");
    bool ok = c.length() == 1 && comando(c[0]);
    server.send(ok ? 200 : 400, "text/plain", ok ? nombre(estado) : "Usar ?c=G, V, R u O");
  });
  server.on("/reconocer", []() { reconocer(); server.send(200, "text/plain", nombre(estado)); });
  server.begin();

  cambiar(TODO_OK);
  Serial.println("ESP32_SAFETY_READY");
}

void loop() {
  static bool conectado = false;
  if ((WiFi.status() == WL_CONNECTED) != conectado) {   // avisar cuando conecta o se cae
    conectado = !conectado;
    if (conectado) { Serial.print("WIFI:"); Serial.println(WiFi.localIP()); }
    mostrar();
  }
  server.handleClient();

  while (Serial.available() > 0) comando(Serial.read());

  if (digitalRead(PIN_BOTON) == LOW) reconocer();

  // Baliza intermitente y sirena con dos tonos mientras dure la alerta
  if (estado == SIN_CASCO || estado == SIN_CHALECO) {
    unsigned long t = millis() - desde;
    digitalWrite(PIN_RELE_BALIZA, (t / 400) % 2 == 0);
    if (estado == SIN_CASCO && !reconocida) sonar((t / 500) % 2 ? 1200 : 1800);
  }
  delay(10);
}
