#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include "arduino_secrets.h"

// --- CONFIGURATION ---
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;
const int HORN_PIN = 5; // D1 on NodeMCU, adjust as needed
const int HORN_DURATION_MS = 800; // 0.8 seconds

// Set your desired Static IP details
IPAddress local_IP(192, 168, 100, 230);
IPAddress gateway(192, 168, 100, 1);      // Your router's IP
IPAddress subnet(255, 255, 255, 0);     // Common subnet mask
IPAddress primaryDNS(8, 8, 8, 8);       // Optional: Google DNS
IPAddress secondaryDNS(8, 8, 4, 4);     // Optional

ESP8266WebServer server(80);

void handleTrigger() {
  Serial.println("Trigger received!");
  digitalWrite(HORN_PIN, HIGH);
  server.send(200, "text/plain", "Horn Triggered");
  delay(HORN_DURATION_MS);
  digitalWrite(HORN_PIN, LOW);
}

void handleRoot() {
  server.send(200, "text/plain", "ESP8266 Horn Trigger is Ready");
}

void setup() {
  Serial.begin(115200);

  // --- STATIC IP CONFIGURATION ---
  // Note: The order for ESP8266 is (IP, Gateway, Subnet, DNS1, DNS2)
  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS)) {
    Serial.println("STA Failed to configure Static IP");
  }

  pinMode(HORN_PIN, OUTPUT);
  digitalWrite(HORN_PIN, LOW);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.print("Static IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.print("ESP8266 MAC Address: ");
  Serial.println(WiFi.macAddress()); // This will print it clearly
  WiFi.setSleepMode(WIFI_NONE_SLEEP);

  server.on("/", handleRoot);
  server.on("/trigger", handleTrigger);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}
