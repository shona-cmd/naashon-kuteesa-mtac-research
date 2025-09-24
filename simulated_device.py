"""
Simulated IoT device: publishes JSON telemetry to MQTT broker.
Run multiple instances with different --device-id to simulate many devices.
"""
import argparse
import json
import random
import time
import paho.mqtt.client as mqtt

BROKER = "broker"  # docker-compose service name; or localhost for local testing
PORT = 1883
TOPIC_PREFIX = "naashonsecureiot/devices"

def main(device_id="device-1", interval=2.0, anomaly_rate=0.01):
    client = mqtt.Client(client_id=f"sim-{device_id}")
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    seq = 0
    try:
        while True:
            seq += 1
            # normal telemetry
            temp = 20 + random.random() * 5
            hum = 30 + random.random() * 10
            # occasionally inject anomaly
            if random.random() < anomaly_rate:
                temp += random.choice([50, -40])  # extreme spike/drop
                anomaly = True
            else:
                anomaly = False
            payload = {
                "device_id": device_id,
                "seq": seq,
                "timestamp": int(time.time()),
                "telemetry": {"temp": temp, "hum": hum},
                "anomaly_injected": anomaly
            }
            client.publish(f"{TOPIC_PREFIX}/{device_id}", json.dumps(payload))
            print(f"[{device_id}] published seq={seq} anomaly={anomaly}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default="device-1")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--anomaly-rate", type=float, default=0.02)
    args = parser.parse_args()
    main(args.device_id, args.interval, args.anomaly_rate)
