import tkinter as tk
import struct
import serial
import threading

SYNC_BYTE = 0xAA
EOF_BYTE = 0xFF
values = [0.0] * 6  # Initialize step values for 6 motors

arduino = serial.Serial('COM4')
arduino.baudrate = 9600
arduino.parity = 'N'
arduino.stopbits = 1
arduino.flush()

def send_command(motor_index, entry):
    try:
        step_value = float(entry.get())
        values = [0.0] * 6  # Initialize step values for 6 motors
        values[motor_index] = step_value * 3200
        message = struct.pack('<BB6fB', SYNC_BYTE, 6, *values, EOF_BYTE)
        arduino.write(message)
        print(f"Sent: {message}")
    except ValueError:
        print(f"Invalid input for Motor {motor_index+1}")

def serial_listener():
    while True:
        if arduino.in_waiting:
            data = arduino.readline().decode().strip()
            print(f"Received: {data}")

threading.Thread(target=serial_listener, daemon=True).start()

root = tk.Tk()
root.title("Stepper Motor Controller")
root.configure(bg="#f0f0f0")
root.geometry("400x300")
root.grid_rowconfigure(tuple(range(8)), weight=1)
root.grid_columnconfigure((0, 1, 2), weight=1)

title_label = tk.Label(root, text="Stepper Motor Control", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
title_label.grid(row=0, column=0, columnspan=3, pady=10, sticky="nsew")

entries = []
for i in range(6):
    tk.Label(root, text=f"Motor {i+1}:", font=("Arial", 12), bg="#f0f0f0").grid(row=i+1, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(root, font=("Arial", 12), justify="center")
    entry.grid(row=i+1, column=1, padx=10, pady=5, sticky="ew")
    entries.append(entry)
    tk.Button(root, text="Step", font=("Arial", 12), bg="#4CAF50", fg="white", command=lambda i=i, e=entry: send_command(i, e)).grid(row=i+1, column=2, padx=10, pady=5, sticky="ew")

tk.Button(root, text="Quit", font=("Arial", 12), bg="#D32F2F", fg="white", command=root.quit).grid(row=7, column=1, pady=10, sticky="ew")

root.mainloop()