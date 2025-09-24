import time


import random

class IoTDevice:
    def __init__(self, device_id):
        self.device_id = device_id

    def generate_data(self):
        # Simulate sensor data
        temperature = random.randint(20, 30)
        humidity = random.randint(40, 60)
        return {
            "device_id": self.device_id,
            "temperature": temperature,

            "humidity": humidity,
        }

    def send_data(self, blockchain):
        data = self.generate_data()
        print(f"Device {self.device_id} sending data: {data}")
        blockchain.add_block(data)


if __name__ == '__main__':
    # Example usage
    from blockchain_layer import Blockchain

    blockchain = Blockchain()
    device1 = IoTDevice("device1")
    device2 = IoTDevice("device2")

    for i in range(3):
        device1.send_data(blockchain)
        device2.send_data(blockchain)
        time.sleep(1)

    print("Blockchain after device data transmission:")
    for block in blockchain.chain:
        print(f"Block #{block.index}:")
        print(f"  Timestamp: {block.timestamp}")
        print(f"  Data: {block.data}")
        print(f"  Hash: {block.hash}")
        print(f"  Previous Hash: {block.previous_hash}")
        print("\\n")
