# Ragnarok mouse firmware

`ragnarok_mouse/ragnarok_mouse.ino` — MCU firmware that turns a serial command
stream from the PC (`src/ragnarok/aim/arduino.py`) into real **USB-HID mouse**
moves/clicks, so input reaches the game below the OS input stack.

## Board
Any **ATmega32u4** board with native USB (Arduino **Leonardo**, **Pro Micro**,
Teensy-class). These enumerate as a real USB mouse via `Mouse.h`. A plain Uno/Nano
(ATmega328) can **not** do native HID — use a 32u4 board (or a USB-Host-Shield
passthrough, not covered by this sketch).

## Protocol
Matches `src/ragnarok/aim/protocol.py` byte-for-byte (verified in CI):
`[0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]`, CRC8 poly 0x07 / init 0x00 over
CMD+LEN+PAYLOAD. Commands: MOVE (dx,dy int16 + button mask), BUTTON (mask),
CONFIG (ignored), DIAG (echoes `micros()` back for HIL latency measurement).

## Flash
1. Arduino IDE → select your 32u4 board + its serial port.
2. Open `ragnarok_mouse/ragnarok_mouse.ino`, Upload.
3. In Ragnarok: **Input tab** → set `mouse_driver = arduino`, `Arduino transport = serial`,
   `Serial port = COMx` (the board's port), `Baud = 115200`.

The board now moves the mouse for the aimbot. Note: for a **single-player offline**
game this hardware path is unnecessary (SendInput works); it exists for setups
that read input at the HID level.
