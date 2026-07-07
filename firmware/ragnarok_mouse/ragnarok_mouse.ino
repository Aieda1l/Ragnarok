// Ragnarok mouse firmware — MAKCU-modeled binary protocol over USB serial.
//
// Board: Arduino Leonardo / Pro Micro / any ATmega32u4 (native USB HID mouse).
// The PC drivers (src/ragnarok/aim/arduino.py + protocol.py) send frames; this
// sketch drives a real USB HID mouse, so moves/clicks reach the game below the
// OS input stack. Must match protocol.py EXACTLY:
//
//   Frame:  [0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]
//   CRC8:   poly 0x07, init 0x00, over CMD + LEN16 + PAYLOAD
//   CMD_MOVE   0x01  payload <hhBB> = dx(i16) dy(i16) buttons(u8) mode(u8)
//   CMD_BUTTON 0x02  payload <B>    = button mask (bit0 L, bit1 R, bit2 M)
//   CMD_CONFIG 0x03  payload = opaque (ignored here)
//   CMD_DIAG   0x04  payload <B> = seq -> echo back CMD_DIAG <I> = micros() (HIL)

#include <Mouse.h>

static const uint8_t START      = 0xAA;
static const uint8_t CMD_MOVE   = 0x01;
static const uint8_t CMD_BUTTON = 0x02;
static const uint8_t CMD_CONFIG = 0x03;
static const uint8_t CMD_DIAG   = 0x04;

// One-byte CRC8 step (poly 0x07, MSB-first) — matches protocol.crc8 byte-by-byte.
static uint8_t crc8_step(uint8_t crc, uint8_t b) {
  crc ^= b;
  for (uint8_t i = 0; i < 8; i++)
    crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  return crc;
}

static uint8_t crc8_buf(const uint8_t* d, uint16_t n) {
  uint8_t c = 0;
  for (uint16_t i = 0; i < n; i++) c = crc8_step(c, d[i]);
  return c;
}

static void setButtons(uint8_t mask) {
  (mask & 0x01) ? Mouse.press(MOUSE_LEFT)   : Mouse.release(MOUSE_LEFT);
  (mask & 0x02) ? Mouse.press(MOUSE_RIGHT)  : Mouse.release(MOUSE_RIGHT);
  (mask & 0x04) ? Mouse.press(MOUSE_MIDDLE) : Mouse.release(MOUSE_MIDDLE);
}

// HID mouse deltas are int8; split a large move into <=127 px steps.
static void applyMove(int16_t dx, int16_t dy, uint8_t buttons) {
  while (dx != 0 || dy != 0) {
    int8_t sx = dx > 127 ? 127 : (dx < -127 ? -127 : (int8_t)dx);
    int8_t sy = dy > 127 ? 127 : (dy < -127 ? -127 : (int8_t)dy);
    Mouse.move(sx, sy, 0);
    dx -= sx;
    dy -= sy;
  }
  setButtons(buttons);
}

static void sendDiagEcho(uint8_t /*seq*/) {
  uint32_t us = micros();
  // body = CMD_DIAG, LEN16_LE(=4), micros LE
  uint8_t body[7] = { CMD_DIAG, 0x04, 0x00,
                      (uint8_t)us, (uint8_t)(us >> 8), (uint8_t)(us >> 16), (uint8_t)(us >> 24) };
  Serial.write(START);
  Serial.write(body, sizeof(body));
  Serial.write(crc8_buf(body, sizeof(body)));
}

// Byte-at-a-time frame parser with a running CRC.
enum { S_START, S_CMD, S_LEN0, S_LEN1, S_PAY, S_CRC };
static uint8_t  state = S_START;
static uint8_t  cmd, lenLo;
static uint16_t len, idx;
static uint8_t  rc;                 // running CRC over CMD+LEN+PAYLOAD
static uint8_t  buf[264];

static void handleFrame() {
  switch (cmd) {
    case CMD_MOVE:
      if (len >= 6) {
        int16_t dx = (int16_t)(buf[0] | (buf[1] << 8));
        int16_t dy = (int16_t)(buf[2] | (buf[3] << 8));
        applyMove(dx, dy, buf[4]);   // buf[5] = mode (unused on this board)
      }
      break;
    case CMD_BUTTON:
      if (len >= 1) setButtons(buf[0]);
      break;
    case CMD_DIAG:
      sendDiagEcho(len >= 1 ? buf[0] : 0);
      break;
    case CMD_CONFIG:
    default:
      break;                         // no-op
  }
}

void setup() {
  Serial.begin(115200);
  Mouse.begin();
}

void loop() {
  while (Serial.available() > 0) {
    uint8_t b = (uint8_t)Serial.read();
    switch (state) {
      case S_START:
        if (b == START) state = S_CMD;
        break;
      case S_CMD:
        cmd = b; rc = crc8_step(0, b); state = S_LEN0;
        break;
      case S_LEN0:
        lenLo = b; rc = crc8_step(rc, b); state = S_LEN1;
        break;
      case S_LEN1:
        len = (uint16_t)lenLo | ((uint16_t)b << 8);
        rc = crc8_step(rc, b);
        idx = 0;
        if (len > sizeof(buf)) { state = S_START; }   // overrun guard
        else                    state = (len ? S_PAY : S_CRC);
        break;
      case S_PAY:
        buf[idx++] = b; rc = crc8_step(rc, b);
        if (idx >= len) state = S_CRC;
        break;
      case S_CRC:
        if (rc == b) handleFrame();                    // CRC ok -> act
        state = S_START;
        break;
    }
  }
}
