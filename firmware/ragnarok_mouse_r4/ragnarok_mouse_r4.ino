// Ragnarok mouse firmware — Arduino UNO R4 (Renesas RA4M1) + USB Host Shield.
//
// TOPOLOGY (passthrough): this board plays TWO USB roles at once —
//   HOST  (USB Host Shield / MAX3421E over SPI/ICSP): reads the REAL mouse's
//         HID reports so your hand motion passes through 1:1.
//   DEVICE(RA4M1 native USB-C): presents ONE HID mouse to the PC = the real
//         passthrough motion PLUS the aim delta Ragnarok injects.
// The game sees a single mouse; there is no separate synthetic input device, and
// no Windows pointer-ballistics in the path.
//
// COMMAND CHANNEL (PC -> this board, aim deltas / clicks), any of:
//   - Serial1 : frames forwarded by the ESP32-S3 over the internal UART (WiFi/UDP
//               path — see firmware/ragnarok_esp32_udp).
//   - vendor HID OUTPUT report : driverless (hidapi) — see the "Vendor HID" note
//               in firmware/README.md (needs a one-line composite-descriptor edit
//               in the Renesas core; until then use Serial1/UDP or USB-CDC serial).
//
// Frame MUST match src/ragnarok/aim/protocol.py EXACTLY (verified in CI):
//   [0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]  CRC8 poly 0x07 init 0x00 over CMD+LEN+PAYLOAD
//   CMD_MOVE 0x01 <hhBB>=dx,dy,buttons,mode   CMD_BUTTON 0x02 <B>=mask
//   CMD_CONFIG 0x03 opaque   CMD_DIAG 0x04 <B>=seq -> echo CMD_DIAG <I>=micros()
//
// Libraries: felis/USB_Host_Shield_2.0 (R4/RA4M1-compatible), Mouse.h (native HID).

#include <SPI.h>
#include <usbhid.h>
#include <hiduniversal.h>
#include <usbhub.h>
#include <Mouse.h>

static const uint8_t START = 0xAA, CMD_MOVE = 0x01, CMD_BUTTON = 0x02,
                     CMD_CONFIG = 0x03, CMD_DIAG = 0x04;

static uint8_t crc8_step(uint8_t crc, uint8_t b) {
  crc ^= b;
  for (uint8_t i = 0; i < 8; i++)
    crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  return crc;
}
static uint8_t crc8_buf(const uint8_t* d, uint16_t n) {
  uint8_t c = 0; for (uint16_t i = 0; i < n; i++) c = crc8_step(c, d[i]); return c;
}

// Injected command state (from the PC over Serial1 / HID). Accumulated between
// loop() iterations and drained once per iteration into a single HID report.
static volatile int32_t inj_dx = 0, inj_dy = 0;
static volatile uint8_t inj_buttons = 0;
// Last real-mouse button mask (updated by the host-shield reader); merged with
// inj_buttons and applied every loop() so injected clicks work on an idle hand.
static volatile uint8_t last_real_buttons = 0;

static void setButtons(uint8_t mask) {
  (mask & 0x01) ? Mouse.press(MOUSE_LEFT)   : Mouse.release(MOUSE_LEFT);
  (mask & 0x02) ? Mouse.press(MOUSE_RIGHT)  : Mouse.release(MOUSE_RIGHT);
  (mask & 0x04) ? Mouse.press(MOUSE_MIDDLE) : Mouse.release(MOUSE_MIDDLE);
}
static void emitMove(int32_t dx, int32_t dy) {        // HID deltas are int8; chunk
  while (dx != 0 || dy != 0) {
    int8_t sx = dx > 127 ? 127 : (dx < -127 ? -127 : (int8_t)dx);
    int8_t sy = dy > 127 ? 127 : (dy < -127 ? -127 : (int8_t)dy);
    Mouse.move(sx, sy, 0);
    dx -= sx; dy -= sy;
  }
}
static void sendDiagEcho() {
  uint32_t us = micros();
  uint8_t body[7] = { CMD_DIAG, 0x04, 0x00,
                      (uint8_t)us, (uint8_t)(us >> 8), (uint8_t)(us >> 16), (uint8_t)(us >> 24) };
  Serial.write(START); Serial.write(body, sizeof(body)); Serial.write(crc8_buf(body, sizeof(body)));
}

static void handleFrame(uint8_t cmd, const uint8_t* buf, uint16_t len) {
  switch (cmd) {
    case CMD_MOVE:
      if (len >= 6) {
        inj_dx += (int16_t)(buf[0] | (buf[1] << 8));
        inj_dy += (int16_t)(buf[2] | (buf[3] << 8));
        inj_buttons = buf[4];
      }
      break;
    case CMD_BUTTON: if (len >= 1) inj_buttons = buf[0]; break;
    case CMD_DIAG:   sendDiagEcho(); break;
    default: break;                                    // CONFIG: no-op
  }
}

// Byte-at-a-time MAKCU parser (one instance per command source).
struct Parser {
  uint8_t state = 0, cmd = 0, lenLo = 0, rc = 0;
  uint16_t len = 0, idx = 0; uint8_t buf[264];
  void feed(uint8_t b) {
    switch (state) {
      case 0: if (b == START) state = 1; break;
      case 1: cmd = b; rc = crc8_step(0, b); state = 2; break;
      case 2: lenLo = b; rc = crc8_step(rc, b); state = 3; break;
      case 3: len = (uint16_t)lenLo | ((uint16_t)b << 8); rc = crc8_step(rc, b);
              idx = 0; state = (len > sizeof(buf)) ? 0 : (len ? 4 : 5); break;
      case 4: buf[idx++] = b; rc = crc8_step(rc, b); if (idx >= len) state = 5; break;
      case 5: if (rc == b) handleFrame(cmd, buf, len); state = 0; break;
    }
  }
};
static Parser serialParser;    // Serial1 (ESP32 link / WiFi path)

// USB Host Shield: read the real mouse and pass its motion/buttons through.
USB Usb;
class RealMouse : public HIDUniversal {
public:
  RealMouse(USB* p) : HIDUniversal(p) {}
protected:
  void ParseHIDData(USBHID*, bool, uint8_t, uint16_t len, uint8_t* buf) {
    // Boot-mouse report layout: [buttons][dx][dy][wheel]. 16-bit / high-res mice
    // differ — adjust to your device's report (see firmware/README.md).
    if (len >= 3) {
      int8_t dx = (int8_t)buf[1], dy = (int8_t)buf[2];
      Mouse.move(dx, dy, (len >= 4) ? (int8_t)buf[3] : 0);   // passthrough real motion
      last_real_buttons = buf[0] & 0x07;                      // remember real buttons (applied in loop)
    }
  }
};
RealMouse realMouse(&Usb);

void setup() {
  Serial.begin(115200);      // native USB CDC (optional debug + DIAG echo)
  Serial1.begin(921600);     // link to the ESP32-S3 (WiFi command path)
  Mouse.begin();
  if (Usb.Init() == -1) { /* Host Shield init failed — check SPI/ICSP wiring */ }
}

void loop() {
  Usb.Task();                                          // pump the real mouse (passthrough)
  while (Serial.available() > 0) serialParser.feed((uint8_t)Serial.read());
  while (Serial1.available() > 0) serialParser.feed((uint8_t)Serial1.read());
  int32_t dx = inj_dx, dy = inj_dy; inj_dx = 0; inj_dy = 0;   // drain injected aim
  if (dx || dy) emitMove(dx, dy);
  // Apply the merged (real | injected) button mask every loop, NOT only when the
  // real mouse reports — otherwise injected clicks never fire on an idle hand and
  // an injected release can stick until the physical mouse next moves.
  static uint8_t applied_mask = 0;
  uint8_t mask = last_real_buttons | inj_buttons;
  if (mask != applied_mask) { setButtons(mask); applied_mask = mask; }
}
