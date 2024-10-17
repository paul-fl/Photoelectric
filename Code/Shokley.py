import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

# Load the data (assuming CSV format)
data = pd.read_csv('Data/Filters.csv')

# Use a for loop to get cleaned data for every colour
cleaned_data = {}
for i in ['r', 'g', 'b', 'y', 'v', 'uv']:  
    cleaned_data[i] = data.dropna(subset=[f'current_{i} pA', f'unc_{i} pA'])

# Define the Shockley diode equation
def shockley_diode(x, A, B, C):
    return A * (np.exp(B * (x - C)) - 1)

# Function to find the zero crossing voltage
def find_zero_crossing(diode_params):
    # Define a function that subtracts 0 from Shockley diode function
    def diode_minus_zero(x):
        A, B, C = diode_params
        return shockley_diode(x, A, B, C)  # The target is when this equals zero
    voltage_at_zero = fsolve(diode_minus_zero, x0=0)[0]  # Initial guess x0 = 0
    return voltage_at_zero

zero_crossing_voltages = {}

# Fit the Shockley diode function to the data for each color
for color, data_cleaned in cleaned_data.items():
    print(f"Processing data for color: {color}")

    # Adjusted initial guesses based on the data
    initial_A = max(data_cleaned[f'current_{color} pA'])  # Starting with max current
    initial_B = 0.01  # A smaller exponential growth rate to try
    initial_C = 0  # Initial guess for the diode's turn-on voltage
    
    # Bounds for the parameters
    bounds = (
        [0, 0, -2],   # Lower bounds for A, B, C
        [np.inf, 1, 2]  # Upper bounds for A, B, C
    )
    
    p0 = [initial_A, initial_B, initial_C]
    
    # Fit the Shockley diode equation to the data
    popt, pcov = curve_fit(shockley_diode, data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'], 
                           p0=p0, bounds=bounds)

    # Print optimized parameters to check if they make sense
    print(f"Optimized parameters for {color} channel: A={popt[0]:.4f}, B={popt[1]:.4f}, C={popt[2]:.4f}")
    
    # Find the voltage where the current crosses zero (i.e., zero crossing)
    voltage_at_zero = find_zero_crossing(popt)
    zero_crossing_voltages[color] = voltage_at_zero
    print(f"Voltage where {color} channel curve crosses zero: {voltage_at_zero:.4f} V")

    # Plot the data with error bars and the Shockley diode fit
    plt.errorbar(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'], 
                 yerr=data_cleaned[f'unc_{color} pA'], fmt='o', label=f'Data ({color.upper()})')
    plt.plot(data_cleaned['voltage V'], shockley_diode(data_cleaned['voltage V'], *popt), 
             label=f'Diode Fit ({color.upper()})', color='red')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (pA)')
    plt.legend()
    plt.show()

# Plot the zero crossing voltages against the frequency of each color
wavelength = {
    'r': 691.797e-9,
    'g': 584.273e-9,
    'b': 438.157e-9,
    'y': 577.302e-9,
    'v': 405.21e-9,
    'uv': 368.11e-9
}

c = 3e8  # Speed of light in m/s
frequency = {color: c / wave for color, wave in wavelength.items()}

# Plot zero-crossing voltage vs. frequency
plt.plot(list(frequency.values()), list(zero_crossing_voltages.values()), 'o')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Zero Crossing Voltage (V)')
plt.show()
