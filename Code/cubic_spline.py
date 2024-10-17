import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

# Load the data (assuming CSV format)
data = pd.read_csv('Data/Filters.csv')

# Use a for loop to get cleaned data for every colour
cleaned_data = {}
for i in ['r', 'g', 'b', 'y', 'v', 'uv']:  
    cleaned_data[i] = data.dropna(subset=[f'current_{i} pA', f'unc_{i} pA'])

# Define the function to find the zero crossing of the spline
def find_zero_crossing_spline(cubic_spline):
    # Define a function that subtracts 0 from the spline(x)
    def spline_minus_zero(x):
        return cubic_spline(x)  # The target is when this equals zero
    voltage_at_zero = fsolve(spline_minus_zero, x0=0)[0]  # Initial guess x0 = 0
    return voltage_at_zero

zero_crossing_voltages = {}

# Fit the cubic spline to the data for each color
for color, data_cleaned in cleaned_data.items():
    print(f"Processing data for color: {color}")

    # Fit a cubic spline
    spline = CubicSpline(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'])

    # Find the zero crossing using the spline
    voltage_at_zero = find_zero_crossing_spline(spline)
    zero_crossing_voltages[color] = voltage_at_zero
    print(f"Voltage where {color} channel curve crosses zero: {voltage_at_zero:.4f} V")

    # Plot the data and the cubic spline fit
    plt.errorbar(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'], yerr=data_cleaned[f'unc_{color} pA'], fmt='o', label=f'Data ({color.upper()})')
    plt.plot(data_cleaned['voltage V'], spline(data_cleaned['voltage V']), label=f'Cubic Spline Fit ({color.upper()})', color='red')
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

manual_stopping = {
    'r': 0.336,
    'g': 0.9363,
    'b': 1.329,
    'y': 0.718,
    'v': 1.4615,
    'uv': 1.577
}

c = 3e8  
frequency = {}
for color, wave in wavelength.items():
    frequency[color] = c / wave

# Take absolute value of zero crossing voltages

for color, voltage in zero_crossing_voltages.items():
    zero_crossing_voltages[color] = abs(voltage)

plt.plot(list(frequency.values()), list(zero_crossing_voltages.values()), 'o', label='Zero Crossing Voltage (Cubic Spline)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Zero Crossing Voltage (V)')
plt.legend()
plt.show()

# plt.plot(list(frequency.values()), list(manual_stopping.values()), 'o', label='Manual Stopping Voltage')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Manual Stopping Voltage (V)')
# plt.legend()
# plt.show()

# Fit a linear function to the zero crossing votlage

# Define the linear function
def linear(x, a, b):
    return a * x + b
 
frequency_values = list(frequency.values())
zero_crossing_voltages = list(zero_crossing_voltages.values())

# Perform the curve fitting
popt, pcov = curve_fit(linear, frequency_values, zero_crossing_voltages)

# Calculate the errors (square root of diagonal elements of covariance matrix)
perr = np.sqrt(np.diag(pcov))

# Plotting
plt.plot(frequency_values, zero_crossing_voltages, 'o', label='Zero Crossing Voltage (Cubic Spline)')
plt.plot(frequency_values, linear(np.array(frequency_values), *popt), label=f'Linear Fit: y={popt[0]:.4f}x + {popt[1]:.4f}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Zero Crossing Voltage (V)')
plt.legend()
plt.show()

# Find the gradient of the linear fit
gradient = popt[0]
print(f"Gradient of the linear fit: {gradient} V/Hz")

# Find Planks constant

h = gradient * 1.6e-19

print(f"Plank's constant: {h} J.s")