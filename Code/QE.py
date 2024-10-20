import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv('Data/QE.csv')

print(df.head())

# Define the wavelength of each color in meters
wavelength = {
    'r': 691.797e-9,   # Red
    'v': 405.21e-9,    # Violet
    'uv': 368.11e-9,   # Ultraviolet
    'y': 577.302e-9,   # Yellow
    'b': 438.157e-9,   # Blue
    'g': 528.273e-9    # Green
}

# Speed of light (m/s)
c = 3e8  

# Calculate the frequency for each color
frequency = {}
for color, wave in wavelength.items():
    frequency[color] = c / wave

# calculate the energy of 

# print("Frequencies for each color:", frequency)

# Planck constant (J*s)
h = 6.63 * 10**-34

# Initialize a list to store the QE values for each row
qe_values = []
frequencies = []

# Loop through each row and calculate QE
for idx, row in df.iterrows():
    color = row['Unnamed: 0']  # Get the color of the current row from the first column
    freq = frequency[color]  # Get the frequency corresponding to that color
    current = row['current pA'] * 10**-12  # Convert current to Amperes (A)
    power = row['power muW'] * 10**-6  # Convert power to Watts (W)
    
    # Calculate QE for the current row
    qe = (current * h * freq) / (power * 1.6E-19)  # Calculate QE
    print(f"Quantum Efficiency for {color} channel: {qe}")
    # Store the QE and corresponding frequency
    qe_values.append(qe)
    frequencies.append(freq)

# Plot the QE results against the frequency of each row
plt.scatter(frequencies, qe_values)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Quantum Efficiency (QE)')
plt.title('Quantum Efficiency vs Frequency')
plt.show()

