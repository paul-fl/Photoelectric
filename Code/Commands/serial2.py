import serial

# Instrument port and timeout settings
COM_PORT = "COM3"  # Adjust the COM port as needed
TIMEOUT = 1  # Timeout duration for serial communication

# SCPI Commands to be sent
commands = [
    "*RST",                # Reset to GPIB defaults
    "SYST:ZCH ON",         # Enable zero check
    "RANG 2e-9",           # Set range to 2nA
    "INIT",                # Trigger reading for zero correction
    "SYST:ZCOR:ACQ",       # Acquire zero correction value
    "SYST:ZCOR ON",        # Enable zero correction
    "RANG:AUTO ON",        # Enable auto range
    "SYST:ZCH OFF",        # Disable zero check
    "READ?"                # Trigger and return a reading
]

class SerialInstrument:
    def __init__(self, port: str, timeout: float | None = 1, **serial_kwargs) -> None:
        self._connection = serial.Serial(
            port=port,
            timeout=timeout,
            write_timeout=timeout,
            **serial_kwargs
        )
        idn = self.query("*IDN?")  # Query identification
        if idn:
            self._idn = idn
            print(f"Connected to {idn}.")
        else:
            self.disconnect()
            raise Exception("Serial Instrument could not be identified.")

    def write(self, command: str) -> None:
        """Send a command to the instrument."""
        command += "\n"  # Add termination character
        self._connection.write(command.encode())

    def query(self, command: str) -> str:
        """Send a command and read the response."""
        self.write(command)
        read_bytes = self._connection.readline()[:-1]  # Remove newline
        return read_bytes.decode()

    def execute_commands(self, command_list) -> None:
        """Execute a list of SCPI commands."""
        for cmd in command_list:
            response = self.query(cmd)
            if "?" in cmd:  # Only print responses to query commands
                print(f"Response to {cmd}: {response}")
            else:
                print(f"Executed: {cmd}")

    def disconnect(self) -> None:
        """Close the serial connection."""
        self._connection.close()


if __name__ == "__main__":
    try:
        # Create connection to instrument
        instrument = SerialInstrument(COM_PORT, TIMEOUT)
        
        # Execute SCPI commands
        instrument.execute_commands(commands)
        
        # Close the connection
        instrument.disconnect()

    except Exception as e:
        print(f"Error: {e}")
