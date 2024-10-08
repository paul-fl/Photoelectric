import serial
import time

COM_PORT = "COM3"  # Replace with your actual COM port
BAUD_RATE = 9600
TIMEOUT = 2

try:
    # Open the serial connection
    connection = serial.Serial(
        port=COM_PORT,
        baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=TIMEOUT,
        rtscts=False,
        dsrdtr=False
    )

    print(f"Connected to: {COM_PORT}")
    time.sleep(2)  # Wait for the connection to establish

    # Clear any existing data in the buffers
    connection.reset_input_buffer()
    connection.reset_output_buffer()
    print("Input and output buffers cleared.")

    # List of commands with different termination characters to try
    commands = [
        "SYST:REM\r",    # Carriage Return termination
        "SYST:REM\n",    # Line Feed termination
        "SYST:REM\r\n",  # Carriage Return + Line Feed termination
    ]

    # Send each command and check for response
    for command in commands:
        print(f"Sending command: {command.encode()}")
        connection.write(command.encode())

        # Allow time for the command to be processed
        time.sleep(1)

        # Read and print any response
        bytes_waiting = connection.in_waiting
        print(f"Bytes waiting after command '{command}': {bytes_waiting}")

        if bytes_waiting > 0:
            response = connection.read(bytes_waiting).decode(errors='ignore').strip()
            print(f"Response: {response}")
        else:
            print(f"No response for command: {command}")

    # Close the connection
    connection.close()
    print("Connection closed.")

except serial.SerialException as se:
    print(f"Serial exception: {se}")
except Exception as e:
    print(f"Error: {e}")
