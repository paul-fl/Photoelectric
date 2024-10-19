import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve, curve_fit
from scipy.stats import norm  # Import norm to fit a Gaussian

# Load the data (assuming CSV format)
data = pd.read_csv('Data/Filters.csv')

# Use a for loop to get cleaned data for every colour
cleaned_data = {}
for i in ['r', 'g', 'b', 'y', 'v', 'uv']:  
    cleaned_data[i] = data.dropna(subset=[f'current_{i} pA', f'unc_{i} pA'])

# Define the function to find the zero crossing of the spline
def find_zero_crossing_spline(cubic_spline):
    def spline_minus_zero(x):
        return cubic_spline(x)  # The target is when this equals zero
    voltage_at_zero = fsolve(spline_minus_zero, x0=0)[0]  # Initial guess x0 = 0
    return voltage_at_zero

zero_crossing_voltages = {}
zero_crossing_errors = {}

# Fit the cubic spline to the data for each color, calculate errors
n_iterations = 10000  # Number of bootstrap iterations

for color, data_cleaned in cleaned_data.items():
    # Fit a cubic spline to the original data
    spline = CubicSpline(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'])
    voltage_at_zero = find_zero_crossing_spline(spline)
    zero_crossing_voltages[color] = voltage_at_zero

    # Monte Carlo approach to estimate the uncertainty in the zero-crossing voltage
    voltage_at_zero_bootstrap = []

    for _ in range(n_iterations):
        # Add random noise to the current based on the uncertainty
        perturbed_current = data_cleaned[f'current_{color} pA'] + np.random.normal(0, data_cleaned[f'unc_{color} pA'])
        spline_perturbed = CubicSpline(data_cleaned['voltage V'], perturbed_current)

        # Find zero-crossing for the perturbed data
        voltage_at_zero_perturbed = find_zero_crossing_spline(spline_perturbed)
        voltage_at_zero_bootstrap.append(voltage_at_zero_perturbed)

    # Estimate the uncertainty in the zero-crossing voltage as the standard deviation
    zero_crossing_errors[color] = np.std(voltage_at_zero_bootstrap)
    
    # Print the result
    print(f"{color.upper()} channel: Zero crossing voltage = {voltage_at_zero:.4f} V, Error = {zero_crossing_errors[color]:.4f} V")

    # Plot the data and the cubic spline fit
    plt.errorbar(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'], yerr=data_cleaned[f'unc_{color} pA'], fmt='o', label=f'Data ({color.upper()})')
    plt.plot(data_cleaned['voltage V'], spline(data_cleaned['voltage V']), label=f'Cubic Spline Fit ({color.upper()})', color='red')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (pA)')
    plt.legend()
    plt.show()

    # Plot the histogram of the bootstrapped stopping voltages
    plt.hist(voltage_at_zero_bootstrap, bins=30, density=True, alpha=0.6, color='g', label='Bootstrapped Stopping Voltages')

    # Fit a Gaussian distribution to the bootstrapped voltages
    mean, std = norm.fit(voltage_at_zero_bootstrap)
    
    # Plot the Gaussian curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit\nMean = {mean:.4f} V\nStd Dev = {std:.4f} V')

    # Plot details
    plt.title(f'Gaussian Fit for Stopping Voltage ({color.upper()})')
    plt.xlabel('Stopping Voltage (V)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Continue with the weighted linear fit...
wavelength = {
    'r': 691.797e-9,
    'g': 528.273e-9,
    'b': 438.157e-9,
    'y': 577.302e-9,
    'v': 405.21e-9,
    'uv': 368.11e-9
}

c = 3e8  # Speed of light (m/s)
frequency = {color: c / wave for color, wave in wavelength.items()}

# Convert to lists for fitting
frequency_values = list(frequency.values())
zero_crossing_voltages_list = list(zero_crossing_voltages.values())
zero_crossing_errors_list = list(zero_crossing_errors.values())

# Define the linear function for fitting
def linear(x, a, b):
    return a * x + b

zero_crossing_voltages_list = np.abs(zero_crossing_voltages_list)

# Perform a weighted curve fitting using the error bars as weights (1 / sigma^2)
popt, pcov = curve_fit(linear, frequency_values, zero_crossing_voltages_list, sigma=zero_crossing_errors_list, absolute_sigma=True)

# Calculate the errors (square root of diagonal elements of covariance matrix)
perr = np.sqrt(np.diag(pcov))

# Plotting
plt.errorbar(frequency_values, zero_crossing_voltages_list, yerr=zero_crossing_errors_list, fmt='o', label='Zero Crossing Voltage (Cubic Spline)')
plt.plot(frequency_values, linear(np.array(frequency_values), *popt), label='Linear Fit')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Zero Crossing Voltage (V)')
plt.legend()
plt.show()

# Find the gradient of the linear fit
gradient = popt[0]
print(f"Gradient of the linear fit: {gradient} V/Hz")



# Calculate Planck's constant using the gradient
h = gradient * 1.6e-19
print(f"Planck's constant: {h} J.s")
print(f"Error in Planck's constant: {perr[0] * 1.6e-19} J.s")
