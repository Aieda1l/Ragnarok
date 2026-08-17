# Ragnarok mouse firmware

Three sketches, for two hardware paths. All speak the same wire protocol as
`src/ragnarok/aim/protocol.py` (verified in CI):
`[0xAA][CMD][LEN16_LE][PAYLOAD][CRC8]`, CRC8 poly 0x07 / init 0x00 over
CMD+LEN+PAYLOAD. Commands: MOVE (dx,dy int16 + button mask), BUTTON (mask),
CONFIG (ignored), DIAG (echoes `micros()` back for HIL latency measurement).

| Sketch | Board | Role |
|---|---|---|
| `ragnarok_mouse_r4/` | UNO R4 (RA4M1) **+ USB Host Shield** | **Passthrough** (recommended for the user's rig) |
| `ragnarok_esp32_udp/` | UNO R4 WiFi's **ESP32-S3** | WiFi/UDP → UART bridge (optional) |
| `ragnarok_mouse/` | ATmega32u4 (Leonardo / Pro Micro) | Standalone HID over USB-CDC serial (legacy, no passthrough) |

## Passthrough topology (UNO R4 + USB Host Shield)

The R4 plays **two USB roles at once**:

```
  Real mouse ─USB─▶ [USB Host Shield / MAX3421E] ─SPI(ICSP)─▶ Arduino R4 (RA4M1)
                                                                 │  merge
  Ragnarok PC ─command frames (HID report / serial / WiFi-UDP)─▶│  passthrough + aim
                                                                 │
  Arduino R4 native USB-C ─ONE combined HID mouse stream─▶ PC / game
```

- **Host role (shield):** `felis/USB_Host_Shield_2.0` (R4/RA4M1-compatible) reads
  your real mouse's HID reports over SPI/ICSP; the sketch relays that motion 1:1.
- **Device role (native USB-C):** the RA4M1 presents one HID mouse to the PC; each
  loop it emits `passthrough + injected aim`. No Windows pointer-ballistics involved.
- **Command channel** (PC → board, aim deltas): pick one —
  - **Serial1 / WiFi-UDP:** the ESP32-S3 bridge (below) forwards frames over the
    internal UART. Set Ragnarok `arduino.transport = udp`, `host` = the bridge IP,
    `udp_port` = `CMD_PORT`.
  - **USB-CDC serial:** if you drive the RA4M1's native CDC directly. Set
    `arduino.transport = serial`, `port = COMx`.
  - **Vendor HID OUTPUT report (driverless, no COM port):** Ragnarok's
    `arduino.transport = hid` + `vid`/`hid_pid` sends commands as HID reports via
    hidapi. This needs a one-line **composite-descriptor edit** in the Renesas core
    (add a vendor collection with an OUTPUT report, usage page 0xFF00) and a
    callback that feeds those report bytes into a second `Parser`. Until you add
    that, use Serial1/UDP or USB-CDC. (The PC side is implemented and unit-tested;
    the descriptor edit is the only box-only piece.)

### Flash (R4 passthrough)
1. Arduino IDE → Boards Manager → install **Arduino UNO R4 Boards (Renesas)**.
2. Library Manager → install **USB Host Shield Library 2.0** (felis).
3. Seat the USB Host Shield; plug your real mouse into the shield's USB-A.
4. Open `ragnarok_mouse_r4/ragnarok_mouse_r4.ino`, select the R4 board + port, Upload.
   - **Native-HID note:** when the RA4M1 takes the USB-C lines for native HID, the
     upload port can disappear; double-tap **RESET** to re-enter the bootloader to
     re-flash.
5. Adjust `RealMouse::ParseHIDData` if your mouse isn't a boot-protocol mouse
   (high-res / 16-bit mice put dx/dy in different bytes — check its HID report).

### MOVE precision
This sketch chunks moves to int8 HID deltas (±127/report), matching the PC
driver. A **16-bit report descriptor** (one move per report, no chunking) is a
future refinement; int8 chunking is correct, just more reports for large flicks.

## ESP32-S3 UDP bridge (optional, WiFi path)

`ragnarok_esp32_udp/` turns the R4 WiFi's ESP32-S3 into a UDP→UART bridge. This
**overwrites the stock USB-serial bridge firmware** on the ESP32-S3.

1. Set `SSID`/`PASS`/`CMD_PORT` in the sketch.
2. Flash the ESP32-S3 per Arduino's *"UNO R4 WiFi — Custom Firmware Upload to
   ESP32"* guide (`espflash`). Keep a copy of the stock bridge to restore later.
3. The bridge prints its IP on boot — set Ragnarok `arduino.host` to it.
4. Wire the ESP32 UART to the RA4M1's `Serial1` at matching baud (921600).

Latency caveat: UDP on the ESP32 bunches packets (~200 ms) — use it for
convenience/config, not for the tight aim loop.

## Legacy: `ragnarok_mouse/` (ATmega32u4)

Standalone HID mouse driven entirely by the PC over USB-CDC serial (no
passthrough). For Leonardo / Pro Micro. Set Ragnarok `arduino.transport = serial`,
`port = COMx`, `baud = 115200`.

## Verify (box-only)
Flash → plug the real mouse into the shield → confirm the OS sees ONE mouse that
moves 1:1 → run Ragnarok with `input.mouse_driver = arduino` and your chosen
transport → enable the trigger → confirm injected aim/clicks reach the game →
`uv run python scripts/measure_hil.py` returns a round-trip latency (DIAG echo).
