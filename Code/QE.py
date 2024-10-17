import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv('Data/QE.csv')
print(df.head())

# Define the wavelength of each color in meters
wavelength = {
    'r': 691.797e-9,
    'g': 528.273e-9,
    'b': 438.157e-9,
    'y': 577.302e-9,
    'v': 405.21e-9,
    'uv': 368.11e-9
}

# Speed of light (m/s)
c = 3e8  

# Calculate the frequency for each color
frequency = {}
for color, wave in wavelength.items():
    frequency[color] = c / wave

# Define a function to calculate Quantum Efficiency (QE)
def QE(data, color_freq):
    return (data['current pA'] * 10**-12 * 6.63 * 10**-34 * color_freq) / (data['power muW'] * 10**-6 * 1.6 * 10**-19)


# Initialize a dictionary to store the QE results for each color
qe_results = {}

# Calculate QE for each color

for color, freq in frequency.items():
    qe_results[color] = QE(df, freq)

# Plot the QE results against the frequency of each color

plt.plot(list(frequency.values()), list(qe_results.values()), 'o')
plt.show()




