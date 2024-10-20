import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

# Load the data (assuming CSV format)
data = pd.read_csv('Data/Filters.csv')

# Use a for loop to get cleaned data for every colour
cleaned_data = {}
for i in ['r', 'g', 'b', 'y', 'v', 'uv']:  
    cleaned_data[i] = data.dropna(subset=[f'current_{i} pA', f'unc_{i} pA'])

# Define color-specific voltage thresholds for flat and rising regions
# Define an upper limit for the rising fit (adjust the upper limit as needed for each color)
voltage_thresholds = {
    'r': {'rising_start': 0.25, 'rising_end': 5.0},  # Rising fit from 2.5V to 5.0V for red
    'g': {'rising_start': -0.45, 'rising_end': 5.5},
    'b': {'rising_start': -1.1, 'rising_end': 6.7},
    'y': {'rising_start': -0.5, 'rising_end': 5.2},
    'v': {'rising_start': -1.2, 'rising_end': 7.8},
    'uv': {'rising_start': -1.6, 'rising_end': 7.6}
}

# Function to find intersection of two lines
def find_intersection(line1, line2):
    # line1 and line2 are tuples of (slope, intercept)
    slope1, intercept1 = line1
    slope2, intercept2 = line2
    # Calculate the intersection point
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    y_intersect = slope1 * x_intersect + intercept1
    return x_intersect, y_intersect

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
    slope_flat, intercept_flat, _, _, _ = linregress(flat_region['voltage V'], flat_region[f'current_{color} pA'])
    
    # Perform linear fit for rising region (only within the specified range)
    slope_rising, intercept_rising, _, _, _ = linregress(rising_region['voltage V'], rising_region[f'current_{color} pA'])
    
    # Find the intersection point (stopping voltage)
    x_intersect, y_intersect = find_intersection((slope_flat, intercept_flat), (slope_rising, intercept_rising))
    print(f"Stopping voltage for {color}: {x_intersect:.4f} V")
    
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
