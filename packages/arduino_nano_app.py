import asyncio
import json
import threading
import time

import serial
from serial.tools import list_ports

from libs.nats.jetstream_manager import JetStreamManager


def get_serial_port():
    ports = list_ports.comports()
    for port in ports:
        if "Nano" in port.description:
            return port.device
    return None


def setup_serial(port, baudrate=9600):
    return serial.Serial(port, baudrate, timeout=1)


def serial_listener(port):
    while True:
        if port.in_waiting:
            data = port.readline().decode().strip()
            print(data)


# For test purposes
def send_data(port):
    while True:
        some_data = json.dumps({"x": 1})
        encoded_data = (some_data + "\n").encode()
        port.write(encoded_data)
        print(f"Sent to Arduino: {some_data}")
        time.sleep(2)


async def main():
    arduino_port = get_serial_port()
    if not arduino_port:
        print("Arduino Nano not connected.  Please check the connection")
        return

    # Setup serial communication
    ser = setup_serial(arduino_port)
    time.sleep(2)

    # Start listening and sending data
    threading.Thread(target=serial_listener, args=(ser,), daemon=True).start()

    # Setup Jetstream
    jsm = JetStreamManager()
    await jsm.connect()

    # Ensure event stream exists
    await jsm.ensure_stream(
        "camera_events",
        subjects=["camera.*"],
        max_msgs=100_000,
    )
    #
    await jsm.ensure_stream("camera_events", subjects=["camera.*"], max_msgs=100_100)

    sub = await jsm.create_consumer(
        "camera.collected",
        "camera_events",
        "camera_processor",
    )

    async for msg in jsm.process_messages(sub, batch_size=25):
        try:
            # Decode and parse JSON
            decoded_data = json.loads(msg.data.decode())

            # Convert to string and send to serial port
            data_str = json.dumps(decoded_data)
            ser.write((data_str + "\n").encode())
            await msg.ack()

        except Exception:
            await msg.nak()


if __name__ == "__main__":
    asyncio.run(main())
