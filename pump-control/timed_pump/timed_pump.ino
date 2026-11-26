const uint8_t PUMP_PIN = 9;
const unsigned long MIN_PULSE_MS = 100;
const unsigned long MAX_PULSE_MS = 2000;

unsigned long startedAt = 0;
unsigned long pulseMs = 0;
char frame[5];
uint8_t length = 0;
bool overflow = false;

void stopPump() {
  digitalWrite(PUMP_PIN, HIGH);
  pulseMs = 0;
}

void handleFrame() {
  if (overflow || length != sizeof(frame) || frame[0] != 'P') {
    Serial.println(F("ERR COMMAND"));
    return;
  }
  unsigned long duration = 0;
  for (uint8_t i = 1; i < sizeof(frame); ++i) {
    if (frame[i] < '0' || frame[i] > '9') {
      Serial.println(F("ERR COMMAND"));
      return;
    }
    duration = duration * 10 + (frame[i] - '0');
  }
  if (duration < MIN_PULSE_MS || duration > MAX_PULSE_MS || pulseMs != 0) {
    Serial.println(F("ERR PULSE"));
    return;
  }
  startedAt = millis();
  pulseMs = duration;
  digitalWrite(PUMP_PIN, LOW);
  Serial.println(F("Pump ON"));
}

void setup() {
  digitalWrite(PUMP_PIN, HIGH);
  pinMode(PUMP_PIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (pulseMs != 0 && millis() - startedAt >= pulseMs) {
    stopPump();
    Serial.println(F("Pump DONE"));
  }

  // Leave room for both the command response and a possible timeout response.
  if (!Serial.available() || Serial.availableForWrite() < 32) {
    return;
  }
  const char input = Serial.read();
  if (input == 'S' || input == '?') {
    length = 0;
    overflow = false;
    if (input == 'S') {
      stopPump();
      Serial.println(F("Pump OFF"));
    } else {
      Serial.println(F("PUMP V1"));
    }
  } else if (input == '\n') {
    handleFrame();
    length = 0;
    overflow = false;
  } else if (length < sizeof(frame)) {
    frame[length++] = input;
  } else {
    overflow = true;
  }
}
