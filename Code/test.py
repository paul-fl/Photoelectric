import serial
import time

COM_PORT = "COM3"
TIMEOUT = 2

try:
    connection = serial.Serial(
        port=COM_PORT, 
        baudrate=9600, 
        bytesize=serial.EIGHTBITS, 
        parity=serial.PARITY_NONE, 
        stopbits=serial.STOPBITS_ONE, 
        timeout=TIMEOUT
    )
    
    
    command = "*IDN?\r\n"  
    for _ in range(5):
        connection.write(command.encode())
        print(f"Command sent: {command}")
        
        time.sleep(1)  
        
        bytes_waiting = connection.in_waiting
        print(f"Bytes waiting: {bytes_waiting}")
        
        if bytes_waiting > 0:
            response = connection.read(bytes_waiting).decode().strip()
            print(f"Response: {response}")
            break
        else:
            print("No response yet.")
        
    connection.close()

except Exception as e:
    print(f"Error: {e}")
