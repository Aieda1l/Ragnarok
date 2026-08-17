// Ragnarok WiFi command bridge — ESP32-S3 (the UNO R4 WiFi radio module).
//
// Receives MAKCU command frames over UDP and forwards the raw bytes to the RA4M1
// over the internal UART (Serial1 on the RA4M1 side). This REPLACES the stock
// USB-serial bridge firmware on the ESP32-S3 — see firmware/README.md for the
// espflash upload procedure (you can restore the stock bridge afterwards).
//
// Honest latency note: ESP32 WiFi UDP tends to bunch packets (~200 ms bursts), so
// this is a convenience / config channel, NOT the low-latency aim path. For
// tight aim use USB (serial or raw-HID).
//
// Frame format is opaque here — bytes are forwarded verbatim; framing/CRC are
// validated on the RA4M1 (see src/ragnarok/aim/protocol.py).

#include <WiFi.h>
#include <WiFiUdp.h>

const char* SSID = "YOUR_SSID";       // <-- set these
const char* PASS = "YOUR_PASS";
const uint16_t CMD_PORT = 9999;       // must match Ragnarok arduino.udp_port

WiFiUDP udp;
uint8_t pkt[512];

void setup() {
  Serial1.begin(921600);              // UART to the RA4M1 (match its Serial1 baud)
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) delay(100);
  udp.begin(CMD_PORT);
  // The RA4M1 needs this board's IP for Ragnarok arduino.host; print it once.
  Serial.begin(115200);
  Serial.print("Ragnarok UDP bridge at ");
  Serial.print(WiFi.localIP());
  Serial.print(":");
  Serial.println(CMD_PORT);
}

void loop() {
  int n = udp.parsePacket();
  if (n > 0) {
    int len = udp.read(pkt, sizeof(pkt));
    if (len > 0) Serial1.write(pkt, len);   // forward MAKCU frame(s) verbatim to the RA4M1
  }
}
