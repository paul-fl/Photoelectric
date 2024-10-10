import serial
import time

# Define the COM port and settings
COM_PORT = "COM3"  # Replace with your actual COM port
BAUD_RATE = 9600
TIMEOUT = 2  # Timeout in seconds

try:
    # Open the serial connection
    connection = serial.Serial(
        port=COM_PORT,
        baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=TIMEOUT,
        rtscts=False,  # Disable RTS/CTS hardware flow control
        dsrdtr=False   # Disable DSR/DTR flow control
    )
    
    print(f"Connected to: {COM_PORT}")
    
    # Allow the connection to establish
    time.sleep(2)

    # Clear the input and output buffers
    connection.reset_input_buffer()
    connection.reset_output_buffer()
    print("Input and output buffers cleared.")
    
    # Command to set the device in remote mode (SYST:REM)
    command = "SYST:REM\r"  # Carriage return termination character
    print(f"Sending command: {command}")
    connection.write(command.encode())

    # Allow some time for the command to be processed
    time.sleep(1)

    # Optionally check for any response
    bytes_waiting = connection.in_waiting
    print(f"Bytes waiting in buffer: {bytes_waiting}")

    if bytes_waiting > 0:
        response = connection.read(bytes_waiting).decode(errors='ignore').strip()
        print(f"Response: {response}")
    else:
        print("No response received from the device, but command should have been sent successfully.")

    # Close the connection
    connection.close()
    print("Connection closed.")

except serial.SerialException as se:
    print(f"Serial exception: {se}")
except Exception as e:
    print(f"Error: {e}")
