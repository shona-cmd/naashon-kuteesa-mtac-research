class NetworkLayer:
    def __init__(self):
        pass

    def secure_communication(self, data):
        # Simulate secure communication using TLS
        print("Simulating secure communication using TLS...")
        return f"Securely transmitted: {data}"

    def apply_zero_trust(self, device_id):
        # Simulate zero-trust authentication
        print(f"Applying zero-trust authentication for device {device_id}...")
        return True

if __name__ == '__main__':
    # Example usage
    network_layer = NetworkLayer()
    device_id = "device1"

    if network_layer.apply_zero_trust(device_id):
        data = "Sensor data"
        secure_data = network_layer.secure_communication(data)
        print(secure_data)
