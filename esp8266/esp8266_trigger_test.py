import requests
import time

# Use the Static IP you reserved
ESP_IP = "192.168.100.230"
url = f"http://{ESP_IP}/trigger"

print(f"🚀 Sending trigger command to {url}...")

try:
    response = requests.get(url, timeout=2)
    if response.status_code == 200:
        print("✅ SUCCESS: I heard the click!")
    else:
        print(f"❌ FAILED: Server responded with {response.status_code}")
except Exception as e:
    print(f"⚠️ ERROR: Could not reach ESP8266. Check your WiFi. ({e})")