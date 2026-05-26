#include <Servo.h>

Servo base, shoulder1, shoulder2, elbow, wrist, gripper;

// ── Speed limiter settings ───────────────────────────────────
const int   STEP_SIZE = 1;   // max degrees moved per tick
const unsigned long INTERVAL = 10; // ms between ticks (~50 °/s)
// ─────────────────────────────────────────────────────────────

// Current (smoothed) positions
int cur[6];

// Target positions
int tgt[6];

// Gripper uses boolean logic — track its target separately
bool gripperClosed = false;

unsigned long lastTick = 0;

void setup() {
  shoulder1.attach(3);
  shoulder2.attach(5);
  base.attach(6);
  elbow.attach(9);
  gripper.attach(11);
  wrist.attach(10);


  Serial.begin(115200);
  Serial.setTimeout(10);
}

// Move one step toward target; returns updated current value
int stepToward(int current, int target) {
  int diff = target - current;
  if (diff == 0) return current;
  int step = constrain(diff, -STEP_SIZE, STEP_SIZE);
  return current + step;
}

void loop() {
  // ── 1. Parse incoming command ────────────────────────────
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    int val[5] = {90, 90, 90, 90, 0};
    String rem = data;

    for (int i = 0; i < 4; i++) {
      int ci = rem.indexOf(',');
      if (ci != -1) {
        val[i] = rem.substring(0, ci).toInt();
        rem = rem.substring(ci + 1);
      }
    }
    val[4] = rem.toInt();

    tgt[0] = val[0];           // base
    tgt[1] = val[1];           // shoulder1
    tgt[2] = 180 - val[1];    // shoulder2 (mirror)
    tgt[3] = val[2];           // elbow
    tgt[4] = val[3];           // wrist
    gripperClosed = (val[4] == 1);
    tgt[5] = gripperClosed ? 85 : 145;
  }

  // ── 2. Step each servo on the interval ──────────────────
  unsigned long now = millis();
  if (now - lastTick >= INTERVAL) {
    lastTick = now;

    cur[0] = stepToward(cur[0], tgt[0]);
    cur[1] = stepToward(cur[1], tgt[1]);
    cur[2] = stepToward(cur[2], tgt[2]);
    cur[3] = stepToward(cur[3], tgt[3]);
    cur[4] = stepToward(cur[4], tgt[4]);
    cur[5] = stepToward(cur[5], tgt[5]);

    base.write(cur[0]);
    shoulder1.write(cur[1]);
    shoulder2.write(cur[2]);
    elbow.write(cur[3]);
    wrist.write(cur[4]);
    gripper.write(cur[5]);
  }
}