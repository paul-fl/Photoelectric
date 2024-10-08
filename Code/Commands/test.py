import serial
import time

COM_PORT = "COM3"  # Replace with your actual COM port
TIMEOUT = 2

try:
    # Open the serial connection
    connection = serial.Serial(
        port=COM_PORT, 
        baudrate=9600, 
        bytesize=serial.EIGHTBITS, 
        parity=serial.PARITY_NONE, 
        stopbits=serial.STOPBITS_ONE, 
        timeout=TIMEOUT
    )
    
    # Set device to remote mode
    print("Sending command to set remote mode...")
    connection.write("SYST:REM\r".encode())  # Set remote mode

    # Wait and then send *IDN?
    time.sleep(2)  # Give time for the command to take effect
    
    command = "*IDN?\r"  # Send the identification query
    print(f"Sending command: {command}")
    connection.write(command.encode())

    time.sleep(1)  # Wait for the response

    bytes_waiting = connection.in_waiting
    print(f"Bytes waiting: {bytes_waiting}")

    if bytes_waiting > 0:
        response = connection.read(bytes_waiting).decode().strip()
        print(f"Response: {response}")
    else:
        print("No response received from the device.")

    # Close the connection
    connection.close()

except Exception as e:
    print(f"Error: {e}")
