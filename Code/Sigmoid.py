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

## Plot the data for each color
# for color, data_cleaned in cleaned_data.items():
#     print(f"Processing data for color: {color}")
    
#     plt.errorbar(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'], yerr=data_cleaned[f'unc_{color} pA'], fmt='o', label=f'Data ({color.upper()})')
#     plt.xlabel('Voltage (V)')
#     plt.ylabel('Current (pA)')
#     plt.legend()
#     plt.show()

# Define the sigmoid function
def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d


def find_zero_crossing(sigmoid_params):
    # Define a function that subtracts 0 from sigmoid(x)
    def sigmoid_minus_zero(x):
        a, b, c, d = sigmoid_params
        return sigmoid(x, a, b, c, d)  # The target is when this equals zero
    voltage_at_zero = fsolve(sigmoid_minus_zero, x0=0)[0]  # Initial guess x0 = 0
    return voltage_at_zero

zero_crossing_voltages = {}

for color, data_cleaned in cleaned_data.items():
    # print(f"Processing data for color: {color}")
    
    # Estimate initial guesses
    a_guess = max(data_cleaned[f'current_{color} pA']) - min(data_cleaned[f'current_{color} pA'])
    midpoint_current = (max(data_cleaned[f'current_{color} pA']) + min(data_cleaned[f'current_{color} pA'])) / 2
    c_guess_index = np.abs(data_cleaned[f'current_{color} pA'] - midpoint_current).argmin()
    c_guess = data_cleaned['voltage V'].iloc[c_guess_index]
    d_guess = min(data_cleaned[f'current_{color} pA'])
    b_guess = 0.1  # Start with a small guess for the slope

    popt, pcov = curve_fit(sigmoid, data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'],
                           p0=[a_guess, b_guess, c_guess, d_guess])

    # Continue with your plotting and zero crossing logic
    voltage_at_zero = find_zero_crossing(popt)
    zero_crossing_voltages[color] = voltage_at_zero
    print(f"Voltage where {color} channel curve crosses zero: {voltage_at_zero:.4f} V")
    
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


plt.plot(list(frequency.values()), list(zero_crossing_voltages.values()), 'o')
plt.show()



# plt.plot(list(frequency.values()), list(manual_stopping.values()), 'o', label='Zero Crossing Voltage')
# plt.show()