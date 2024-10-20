import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit

# Load the data (assuming CSV format)
data = pd.read_csv('Data/Filters.csv')

# Use a for loop to get cleaned data for every colour
cleaned_data = {}
for i in ['r', 'g', 'b', 'y', 'v', 'uv']:  
    cleaned_data[i] = data.dropna(subset=[f'current_{i} pA', f'unc_{i} pA'])

# Define color-specific voltage thresholds for flat and rising regions
voltage_thresholds = {
    'r': {'rising_start': 0.25, 'rising_end': 5.0},
    'g': {'rising_start': -0.45, 'rising_end': 5.5},
    'b': {'rising_start': -1.1, 'rising_end': 6.7},
    'y': {'rising_start': -0.5, 'rising_end': 5.2},
    'v': {'rising_start': -1.2, 'rising_end': 7.8},
    'uv': {'rising_start': -1.6, 'rising_end': 7.6}
}

wavelength = {
    'r': 691.797e-9,
    'g': 528.273e-9,
    'b': 438.157e-9,
    'y': 577.302e-9,
    'v': 405.21e-9,
    'uv': 368.11e-9
}

c = 3e8  
frequency = {color: c / wave for color, wave in wavelength.items()}

# Function to find intersection of two lines
def find_intersection(line1, line2):
    slope1, intercept1 = line1
    slope2, intercept2 = line2
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    return x_intersect

# Function to find the error in the intersection point
def find_intersection_error(slope1, intercept1, slope2, intercept2, intercept1_error, intercept2_error, slope1_error, slope2_error):
    x_intersect_error = np.sqrt((intercept1_error**2 + intercept2_error**2) / (slope1 - slope2)**2 +
                                ((intercept1 - intercept2)**2 * (slope1_error**2 + slope2_error**2)) / (slope1 - slope2)**4)
    return x_intersect_error

# Dictionary to store intersection voltages and errors for each color
zero_crossing_voltages = {}
zero_crossing_errors = {}

# Plot and linear fits for each color
for color, data_cleaned in cleaned_data.items():
    print(f"Processing data for color: {color}")
    
    # Get the thresholds for the current color
    thresholds = voltage_thresholds[color]
    
    # Select the lowest voltage value (flat start) in the dataset
    flat_start = data_cleaned['voltage V'].min()
    
    # Define a small offset for both flat and rising regions to overlap slightly
    small_offset = 0.25  # This offset will be applied to extend both fits slightly
    
    # Select data for the flat region (from flat_start to just beyond rising_start)
    flat_region = data_cleaned[(data_cleaned['voltage V'] >= flat_start) & 
                               (data_cleaned['voltage V'] <= thresholds['rising_start'] + small_offset)]
    
    # Select data for the rising region within the user-defined range (rising_start - small_offset to rising_end)
    rising_region = data_cleaned[(data_cleaned['voltage V'] >= thresholds['rising_start'] - small_offset) & 
                                 (data_cleaned['voltage V'] <= thresholds['rising_end'])]
    
    # Perform linear fit for flat region
    slope_flat, intercept_flat, r_value_flat, p_value_flat, std_err_flat = linregress(flat_region['voltage V'], flat_region[f'current_{color} pA'])
    
    # Perform linear fit for rising region
    slope_rising, intercept_rising, r_value_rising, p_value_rising, std_err_rising = linregress(rising_region['voltage V'], rising_region[f'current_{color} pA'])
    
    # Calculate error in intercepts (standard errors)
    intercept_flat_error = std_err_flat * np.sqrt(np.mean(flat_region['voltage V']**2))
    intercept_rising_error = std_err_rising * np.sqrt(np.mean(rising_region['voltage V']**2))
    
    # Calculate the intersection point (stopping voltage) and its error
    x_intersect = find_intersection((slope_flat, intercept_flat), (slope_rising, intercept_rising))
    x_intersect_error = find_intersection_error(slope_flat, intercept_flat, slope_rising, intercept_rising, intercept_flat_error, intercept_rising_error, std_err_flat, std_err_rising)
    
    zero_crossing_voltages[color] = x_intersect  # Store intersection voltage
    zero_crossing_errors[color] = x_intersect_error  # Store intersection voltage error
    print(f"Stopping voltage for {color}: {x_intersect:.4f} V with error: {x_intersect_error:.4f} V")
    
    # Plot data points
    plt.errorbar(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'], 
                 yerr=data_cleaned[f'unc_{color} pA'], fmt='o', label=f'Data ({color.upper()})')
    
    # Plot linear fit for the flat region (from flat_start to rising_start + small_offset)
    flat_line_x = np.linspace(flat_start, thresholds['rising_start'] + small_offset, 100)
    plt.plot(flat_line_x, slope_flat * flat_line_x + intercept_flat, 
             label=f'Flat Fit ({color.upper()})', color='blue')
    
    # Plot linear fit for the rising region (starting slightly before rising_start)
    rising_line_x = np.linspace(thresholds['rising_start'] - small_offset, thresholds['rising_end'], 100)
    plt.plot(rising_line_x, slope_rising * rising_line_x + intercept_rising, 
             label=f'Rising Fit ({color.upper()})', color='green')
    
    # Mark the intersection point
    plt.axvline(x=x_intersect, color='red', linestyle='--', label=f'Stopping Voltage ({x_intersect:.4f} V)')
    
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (pA)')
    plt.legend()
    plt.show()

# Take absolute value of zero crossing voltages and errors
for color in zero_crossing_voltages:
    zero_crossing_voltages[color] = abs(zero_crossing_voltages[color])
    zero_crossing_errors[color] = abs(zero_crossing_errors[color])

# Plot the zero crossing voltages against the frequency of each color with error bars
plt.errorbar(list(frequency.values()), list(zero_crossing_voltages.values()), yerr=list(zero_crossing_errors.values()), fmt='o', label='Zero Crossing Voltage')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Zero Crossing Voltage (V)')
plt.legend()
plt.show()

# Fit a linear function to the zero crossing voltage

# Define the linear function
def linear(x, a, b):
    return a * x + b

# Convert to lists for fitting
frequency_values = list(frequency.values())
zero_crossing_voltages_list = list(zero_crossing_voltages.values())

# Fit the linear function and get covariance matrix
popt, pcov = curve_fit(linear, frequency_values, zero_crossing_voltages_list, sigma=list(zero_crossing_errors.values()))

# Calculate errors in fit parameters
perr = np.sqrt(np.diag(pcov))

# Plot the linear fit with error bars
plt.errorbar(frequency_values, zero_crossing_voltages_list, yerr=list(zero_crossing_errors.values()), fmt='o', label='Zero Crossing Voltage')
plt.plot(frequency_values, linear(np.array(frequency_values), *popt), label=f'Linear Fit ($\pm$ {perr[0]:.4f})')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Zero Crossing Voltage (V)')
plt.legend()

plt.show()

# Find the slope and its error
slope = popt[0]
slope_error = perr[0]

# Calculate Planck's constant (h)
h = slope * 1.6e-19
h_error = slope_error * 1.6e-19

print(f"The value of Planck's constant is: {h:.4e} ± {h_error:.4e} J·s")
