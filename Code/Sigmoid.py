import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, fsolve

# Load the data (assuming CSV format)
data = pd.read_csv('Data/Filters.csv')

# Use a for loop to get cleaned data for every colour
cleaned_data = {}
for i in ['r', 'g', 'b', 'y', 'v', 'uv']:  
    cleaned_data[i] = data.dropna(subset=[f'current_{i} pA', f'unc_{i} pA'])

# Define the sigmoid function
def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

# Function to find the zero crossing point
def find_zero_crossing(sigmoid_params):
    def sigmoid_minus_zero(x):
        a, b, c, d = sigmoid_params
        return sigmoid(x, a, b, c, d)  # The target is when this equals zero
    voltage_at_zero = fsolve(sigmoid_minus_zero, x0=0)[0]  # Initial guess x0 = 0
    return voltage_at_zero

# Propagate the error for the zero crossing voltage
def propagate_error(sigmoid_params, pcov):
    a, b, c, d = sigmoid_params
    # The error in the zero crossing is related to the uncertainty in 'c' (the horizontal shift)
    c_error = np.sqrt(pcov[2, 2])  # The error in parameter 'c'
    return c_error

zero_crossing_voltages = {}
zero_crossing_errors = {}

for color, data_cleaned in cleaned_data.items():
    # Estimate initial guesses
    a_guess = max(data_cleaned[f'current_{color} pA']) - min(data_cleaned[f'current_{color} pA'])
    midpoint_current = (max(data_cleaned[f'current_{color} pA']) + min(data_cleaned[f'current_{color} pA'])) / 2
    c_guess_index = np.abs(data_cleaned[f'current_{color} pA'] - midpoint_current).argmin()
    c_guess = data_cleaned['voltage V'].iloc[c_guess_index]
    d_guess = min(data_cleaned[f'current_{color} pA'])
    b_guess = 0.1  # Start with a small guess for the slope

    # Fit the sigmoid function
    popt, pcov = curve_fit(sigmoid, data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'],
                           p0=[a_guess, b_guess, c_guess, d_guess])

    # Find the zero crossing voltage
    voltage_at_zero = find_zero_crossing(popt)
    zero_crossing_voltages[color] = voltage_at_zero

    # Propagate the error in the zero crossing voltage
    voltage_error = propagate_error(popt, pcov)
    zero_crossing_errors[color] = voltage_error

    print(f"Voltage where {color} channel curve crosses zero: {voltage_at_zero:.4f} V ± {voltage_error:.4f} V")
    
    # Plot the data and the sigmoid fit
    plt.errorbar(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'], yerr=data_cleaned[f'unc_{color} pA'], fmt='o')
    plt.plot(data_cleaned['voltage V'], sigmoid(data_cleaned['voltage V'], *popt), label=f'Sigmoid Fit ({color.upper()})', color='red')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (pA)')
    plt.legend()
    plt.show()

# Plot the zero crossing voltages against the frequency of each color
wavelength = {
    'r': 691.797e-9,
    'g': 528.273e-9,
    'b': 438.157e-9,
    'y': 577.302e-9,
    'v': 405.21e-9,
    'uv': 368.11e-9
}

c = 3e8  
frequency = {}
for color, wave in wavelength.items():
    frequency[color] = c / wave

# Take absolute value of zero crossing voltages
for color, voltage in zero_crossing_voltages.items():
    zero_crossing_voltages[color] = abs(voltage)

# Convert the frequency and zero crossing voltages to lists
frequency_values = list(frequency.values())
zero_crossing_voltages_list = list(zero_crossing_voltages.values())
zero_crossing_errors_list = list(zero_crossing_errors.values())

# Plot zero crossing voltages with error bars
plt.errorbar(frequency_values, zero_crossing_voltages_list, 
             yerr=zero_crossing_errors_list, fmt='o', label='Zero Crossing Voltage')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Zero Crossing Voltage (V)')
plt.legend()
plt.show()

# Fit a linear function to the zero crossing voltage

# Define the linear function
def linear(x, a, b):
    return a * x + b

# Perform the linear fit using the error in the zero crossing voltages as weights
popt, pcov = curve_fit(linear, frequency_values, zero_crossing_voltages_list, sigma=zero_crossing_errors_list)

# Get the fit parameters and their errors
slope, intercept = popt
slope_error, intercept_error = np.sqrt(np.diag(pcov))

# Plot the linear fit with error bars
plt.errorbar(frequency_values, zero_crossing_voltages_list, 
             yerr=zero_crossing_errors_list, fmt='o', label='Zero Crossing Voltage')
plt.plot(frequency_values, linear(np.array(frequency_values), *popt), label=f'Linear Fit (slope: {slope:.4e} ± {slope_error:.4e})')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Zero Crossing Voltage (V)')
plt.legend()
plt.show()

# Calculate Planck's constant (h) from the slope of the linear fit
# Planck's constant h = slope * elementary charge (1.6e-19 C)
elementary_charge = 1.6e-19
h = slope * elementary_charge
h_error = slope_error * elementary_charge

print(f"The value of Planck's constant is: {h:.4e} ± {h_error:.4e} J·s")
