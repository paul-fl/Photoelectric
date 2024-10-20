import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Load the data for each color in a for loop
data_dict = {}

for i in ['Blue', 'Green', 'Red', 'Yellow']:
    data_dict[i] = pd.read_csv('Data/Hyperfine/{}.csv'.format(i))

# Define the wavelength dictionary
wavelength = {
    'b': 438.157e-9,
    'g': 528.273e-9,
    'r': 691.797e-9,
    'y': 577.302e-9
}

c = 3e8  
frequency = {color: c / wave for color, wave in wavelength.items()}

# Define the Shockley diode function
def shockley_diode(x, A, B, C):
    return A * (np.exp(B * (x - C)) - 1)

# Plot the data for each color
for color, data in data_dict.items():
    voltage = data['voltage V']
    current = data['current pA']
    unc = data['unc pA']
    current = current * 1e-12
    unc = unc * 1e-12
    
    popt, pcov = curve_fit(shockley_diode, voltage, current, p0=[max(current), 1, 0])
    plt.errorbar(voltage, current, yerr=unc, fmt='x')
    plt.title(f'{color} data')
    plt.plot(voltage, shockley_diode(voltage, *popt))
    plt.show()

    print(f"The value of A, B, C for {color} data is: {popt}")

# Find the zero crossing voltage for each color using the C parameter
zero_crossing_voltages = {}

for color, data in data_dict.items():
    popt, pcov = curve_fit(shockley_diode, data['voltage V'], data['current pA'], p0=[max(data['current pA']), 1, 0])
    zero_crossing_voltages[color] = popt[2]

# Find the error in the zero crossing voltage
zero_crossing_errors = {}

for color, data in data_dict.items():
    popt, pcov = curve_fit(shockley_diode, data['voltage V'], data['current pA'], p0=[max(data['current pA']), 1, 0])
    zero_crossing_errors[color] = np.sqrt(pcov[2, 2])

# Take absolute value of zero crossing voltage
for color, voltage in zero_crossing_voltages.items():
    zero_crossing_voltages[color] = abs(voltage)

# Define the linear function for fitting
def linear(x, a, b):
    return a * x + b

frequency_values = list(frequency.values())
zero_crossing_voltages_list = list(zero_crossing_voltages.values())
zero_crossing_errors_list = list(zero_crossing_errors.values())

# Perform the curve fitting
popt, pcov = curve_fit(linear, frequency_values, zero_crossing_voltages_list, sigma=list(zero_crossing_errors.values()))

# Convert frequency_values to NumPy array for plotting
frequency_values_np = np.array(frequency_values)

# Plot the zero crossing voltages against the frequency of each color
plt.errorbar(frequency_values, zero_crossing_voltages_list, yerr=zero_crossing_errors_list, fmt='o', label='Data')
plt.plot(frequency_values_np, linear(frequency_values_np, *popt), label='Linear Fit')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Zero Crossing Voltage (V)')
plt.title('Zero Crossing Voltage vs Frequency')
plt.legend()
plt.show()

# Find the value of Planck's constant (h)
slope = popt[0]
slope_error = np.sqrt(pcov[0, 0])

h = slope * 1.6e-19
h_error = slope_error * 1.6e-19

print(f"The value of Planck's constant is: {h:.4e} ± {h_error:.4e} J·s")
