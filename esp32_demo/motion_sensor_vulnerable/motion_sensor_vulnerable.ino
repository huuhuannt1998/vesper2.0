/**
 * VESPER ESP32 M5Stack Motion Sensor (Vulnerable Demo)
 * 
 * This firmware implements the VESPER motion sensor protocol on physical ESP32 hardware
 * with the SAME intentional vulnerabilities as the QEMU version for attack demonstration.
 * 
 * Hardware: M5Stack Basic Development Kit v2.7
 * Connection: WiFi (connects to your network, exposes TCP server on port 15011)
 * 
 * DEMO ATTACKS:
 *   1. Buffer Overflow (attack_buffer_overflow_setid) - Overflows device_id buffer
 *   2. Info Disclosure (attack_info_disclosure_debug_dump) - Leaks auth token/memory
 *   3. Auth Bypass (attack_auth_bypass_no_token) - No authentication required
 */

#include <WiFi.h>
#include <M5Unified.h>

// ============================================================================
// WiFi Configuration
// ============================================================================
const char* WIFI_SSID = "SpectrumSetup-25";      // ⚠️ CHANGE THIS
const char* WIFI_PASSWORD = "coffeelotus163";  // ⚠️ CHANGE THIS
const int TCP_PORT = 15011;

// ============================================================================
// Device State Variables
// ============================================================================
uint8_t motion_detected = 0;
uint8_t armed = 1;
uint8_t sensitivity = 5;  // 1-10
uint32_t cooldown_ticks = 0;
uint32_t cooldown_period = 300;
uint32_t detection_count = 0;
uint32_t tick_count = 0;
uint32_t rand_seed = 54321;

// ⚠️ VULNERABILITY: Fixed-size buffers - device_id and auth_token are adjacent in memory
// Overflowing device_id[16] corrupts auth_token which immediately follows it
char device_id[16] = "esp32-motion-001";  // Only 16 bytes!
char auth_token[64] = "SECRET_KEY_9x7z";  // Pre-seeded so corruption is visible

WiFiServer server(TCP_PORT);
WiFiClient client;

// ============================================================================
// Helper Functions
// ============================================================================

void uart_puts(const char* s) {
  if (client && client.connected()) {
    client.print(s);
  }
  Serial.print(s);
}

void uart_put_int(int n) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%d", n);
  uart_puts(buf);
}

uint32_t simple_rand() {
  rand_seed = rand_seed * 1103515245 + 12345;
  return (rand_seed >> 16) & 0x7FFF;
}

void simulate_motion() {
  if (!armed) return;
  if (cooldown_ticks > 0) {
    cooldown_ticks--;
    return;
  }
  
  // Use M5Stack accelerometer for real motion detection
  auto imu_update = M5.Imu.update();
  if (imu_update) {
    auto data = M5.Imu.getImuData();
    float accX = data.accel.x;
    float accY = data.accel.y;
    float accZ = data.accel.z;
    
    // Threshold-based detection (shake or tilt triggers motion)
    if (abs(accX) > 0.3 || abs(accY) > 0.3 || abs(accZ - 1.0) > 0.3) {
      motion_detected = 1;
      detection_count++;
      cooldown_ticks = cooldown_period;
    }
  }
}

// ⚠️ VULNERABILITY: str_copy has no bounds checking - buffer overflow possible
void str_copy(char* dst, const char* src, int max) {
  int i = 0;
  while (src[i] && i < max - 1) {
    dst[i] = src[i];
    i++;
  }
  dst[i] = '\0';
  // BUG: If src is longer than max, this still writes max-1 chars
  // but the overflow happens in the calling code that doesn't check length
}

// ============================================================================
// Command Processing (VESPER Protocol)
// ============================================================================

void process_command(char* cmd) {
  // LED feedback on M5Stack
  M5.Display.fillRect(0, 220, 320, 20, TFT_BLACK);
  M5.Display.setCursor(0, 220);
  M5.Display.setTextColor(TFT_YELLOW);
  M5.Display.printf("CMD: %s", cmd);
  
  if (strcmp(cmd, "GET_MOTION") == 0) {
    uart_puts("MOTION:");
    uart_puts(motion_detected ? "active" : "inactive");
    uart_puts("\n");
    motion_detected = 0;
  }
  else if (strcmp(cmd, "GET_COUNT") == 0) {
    uart_puts("COUNT:");
    uart_put_int(detection_count);
    uart_puts("\n");
  }
  else if (strcmp(cmd, "ARM") == 0) {
    armed = 1;
    uart_puts("ARMED:yes\nACK\n");
    M5.Display.fillRect(0, 60, 320, 40, TFT_GREEN);
    M5.Display.setCursor(10, 70);
    M5.Display.setTextColor(TFT_BLACK);
    M5.Display.setTextSize(3);
    M5.Display.print("ARMED");
  }
  else if (strcmp(cmd, "DISARM") == 0) {
    armed = 0;
    uart_puts("ARMED:no\nACK\n");
    M5.Display.fillRect(0, 60, 320, 40, TFT_RED);
    M5.Display.setCursor(10, 70);
    M5.Display.setTextColor(TFT_WHITE);
    M5.Display.setTextSize(3);
    M5.Display.print("DISARMED");
  }
  else if (strcmp(cmd, "GET_ARMED") == 0) {
    uart_puts("ARMED:");
    uart_puts(armed ? "yes" : "no");
    uart_puts("\n");
  }
  else if (strncmp(cmd, "SET_SENSITIVITY:", 16) == 0) {
    int val = cmd[16] - '0';
    if (val >= 1 && val <= 9) {
      sensitivity = val;
    }
    uart_puts("SENSITIVITY:");
    uart_put_int(sensitivity);
    uart_puts("\nACK\n");
  }
  else if (strcmp(cmd, "STATUS") == 0) {
    uart_puts("STATUS:OK\n");
  }
  else if (strcmp(cmd, "IDENTIFY") == 0 || strcmp(cmd, "ID") == 0) {
    uart_puts("DEVICE:VESPER_ESP32_MOTION\nTYPE:MOTION_SENSOR\n");
    uart_puts("CAPS:motion,armed,sensitivity\nFIRMWARE:1.0.0-ESP32\n");
    uart_puts("ID:");
    uart_puts(device_id);
    uart_puts("\n");
  }
  // ⚠️ VULNERABILITY 1: Buffer Overflow in SET_ID (no length check)
  else if (strncmp(cmd, "SET_ID:", 7) == 0) {
    // INTENTIONALLY VULNERABLE: No bounds checking!
    // device_id is only 16 bytes; auth_token immediately follows in memory
    // Sending >16 chars corrupts auth_token
    char* payload = cmd + 7;
    int payload_len = strlen(payload);
    strcpy(device_id, payload);  // UNSAFE! Overflows into auth_token if >15 chars
    uart_puts("ID:");
    uart_puts(device_id);
    uart_puts("\n");
    // Report whether overflow reached auth_token
    if (payload_len > 15) {
      uart_puts("OVERFLOW:DETECTED\n");
      uart_puts("CORRUPTED_TOKEN:");
      uart_puts(auth_token);  // Will show corrupted data
      uart_puts("\nACK\n");
      // Visual indicator on LCD
      M5.Display.fillRect(0, 60, 320, 40, TFT_PURPLE);
      M5.Display.setCursor(10, 70);
      M5.Display.setTextColor(TFT_WHITE);
      M5.Display.setTextSize(2);
      M5.Display.print("OVERFLOW!");
    } else {
      uart_puts("ACK\n");
    }
  }
  // ⚠️ VULNERABILITY 2: Authentication Bypass (always accepts)
  else if (strncmp(cmd, "AUTH:", 5) == 0) {
    str_copy(auth_token, cmd + 5, 64);
    uart_puts("AUTH:OK\nACK\n");  // Always succeeds - no validation!
  }
  // ⚠️ VULNERABILITY 3: Information Disclosure (debug backdoor)
  else if (strcmp(cmd, "DEBUG_DUMP") == 0) {
    uart_puts("DEBUG:MEMORY_DUMP\n");
    uart_puts("SEED:");
    uart_put_int(rand_seed);
    uart_puts("\n");
    uart_puts("TOKEN:");
    uart_puts(auth_token);  // Leaks secret token!
    uart_puts("\n");
    uart_puts("TICKS:");
    uart_put_int(tick_count);
    uart_puts("\n");
    uart_puts("WIFI_IP:");
    uart_puts(WiFi.localIP().toString().c_str());
    uart_puts("\n");
  }
  // ON/OFF aliases for ARM/DISARM (compatibility with attack framework)
  else if (strcmp(cmd, "ON") == 0) {
    armed = 1;
    uart_puts("SWITCH:on\nACK\n");
    M5.Display.fillRect(0, 60, 320, 40, TFT_GREEN);
    M5.Display.setCursor(10, 70);
    M5.Display.setTextColor(TFT_BLACK);
    M5.Display.setTextSize(3);
    M5.Display.print("ARMED");
  }
  else if (strcmp(cmd, "OFF") == 0) {
    armed = 0;
    uart_puts("SWITCH:off\nACK\n");
    M5.Display.fillRect(0, 60, 320, 40, TFT_RED);
    M5.Display.setCursor(10, 70);
    M5.Display.setTextColor(TFT_WHITE);
    M5.Display.setTextSize(3);
    M5.Display.print("DISARMED");
  }
  else if (strcmp(cmd, "GET_SWITCH") == 0) {
    uart_puts("SWITCH:");
    uart_puts(armed ? "on" : "off");
    uart_puts("\n");
  }
  else if (strcmp(cmd, "REBOOT") == 0) {
    uart_puts("ACK:REBOOT\n");
    delay(100);
    ESP.restart();
  }
  else if (cmd[0] != '\0') {
    uart_puts("ERROR:UNKNOWN:");
    uart_puts(cmd);
    uart_puts("\n");
  }
}

// ============================================================================
// Setup & Loop
// ============================================================================

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  
  // Setup LCD
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setTextColor(TFT_WHITE);
  M5.Display.setTextSize(2);
  M5.Display.setCursor(10, 10);
  M5.Display.println("VESPER ESP32 Sensor");
  M5.Display.println("Vulnerable Demo");
  
  Serial.begin(115200);
  Serial.println("\nVESPER ESP32 Motion Sensor (Vulnerable)");
  Serial.println("BOOTED");
  
  // Connect to WiFi
  M5.Display.setCursor(10, 60);
  M5.Display.print("WiFi: ");
  M5.Display.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int dots = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    M5.Display.print(".");
    dots++;
    if (dots > 20) {
      M5.Display.setCursor(10, 90);
      M5.Display.setTextColor(TFT_RED);
      M5.Display.println("WiFi Failed!");
      M5.Display.println("Check SSID/Password");
      while(1) delay(1000);
    }
  }
  
  M5.Display.setCursor(10, 90);
  M5.Display.setTextColor(TFT_GREEN);
  M5.Display.print("IP: ");
  M5.Display.println(WiFi.localIP());
  
  M5.Display.setCursor(10, 110);
  M5.Display.setTextColor(TFT_CYAN);
  M5.Display.print("Port: ");
  M5.Display.println(TCP_PORT);
  
  Serial.print("WiFi connected: ");
  Serial.println(WiFi.localIP());
  Serial.print("TCP server on port: ");
  Serial.println(TCP_PORT);
  
  // Start TCP server
  server.begin();
  
  M5.Display.setCursor(10, 140);
  M5.Display.setTextColor(TFT_WHITE);
  M5.Display.println("Ready for attacks!");
  M5.Display.println("Connect from:");
  M5.Display.print("  ");
  M5.Display.println(WiFi.localIP().toString() + ":15011");
  
  uart_puts("DEVICE:VESPER_ESP32_MOTION\nREADY\n");
}

void loop() {
  M5.update();
  
  // Handle new client connections
  if (!client || !client.connected()) {
    client = server.available();
    if (client) {
      Serial.println("Client connected!");
      M5.Display.fillRect(0, 200, 320, 20, TFT_GREEN);
      M5.Display.setCursor(10, 200);
      M5.Display.setTextColor(TFT_BLACK);
      M5.Display.print("CLIENT CONNECTED");
      client.println("BOOTED\nDEVICE:VESPER_ESP32_MOTION\nREADY");
    }
  }
  
  // Process incoming commands
  if (client && client.connected() && client.available()) {
    String line = client.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      char cmd[256];
      line.toCharArray(cmd, sizeof(cmd));
      process_command(cmd);
    }
  }
  
  // Simulate motion detection
  tick_count++;
  if (tick_count % 50 == 0) {
    simulate_motion();
  }
  
  // Send motion events
  if (motion_detected) {
    uart_puts("EVENT:MOTION:active\n");
    M5.Display.fillRect(0, 180, 320, 20, TFT_ORANGE);
    M5.Display.setCursor(10, 180);
    M5.Display.setTextColor(TFT_BLACK);
    M5.Display.print("MOTION DETECTED!");
    motion_detected = 0;
    delay(100);
  }
  
  delay(20);
}
