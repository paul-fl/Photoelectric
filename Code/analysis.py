import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Load the data (assuming CSV format)
data = pd.read_csv('Data/Filters.csv')

# Use a for loop to get cleaned data for every colour
cleaned_data = {}
for i in ['r', 'g', 'b', 'y', 'v', 'uv']:  
    cleaned_data[i] = data.dropna(subset=[f'current_{i} pA', f'unc_{i} pA'])

for color, data_cleaned in cleaned_data.items():
    print(f"Processing data for color: {color}")
    
    plt.errorbar(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'], yerr=data_cleaned[f'unc_{color} pA'], fmt='o', label=f'Data ({color.upper()})')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (pA)')
    plt.legend()
    plt.show()

def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

for color, data_cleaned in cleaned_data.items():
    print(f"Processing data for color: {color}")

    popt, pcov = curve_fit(sigmoid, data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'],
        p0=[max(data_cleaned[f'current_{color} pA']), 1, 0, min(data_cleaned[f'current_{color} pA'])])
    print(f"Optimized parameters for {color} channel:", popt)
    print("Covariance of parameters:", pcov)

    plt.errorbar(data_cleaned['voltage V'], data_cleaned[f'current_{color} pA'], yerr=data_cleaned[f'unc_{color} pA'], fmt='o', label=f'Data ({color.upper()})')
    plt.plot(data_cleaned['voltage V'], sigmoid(data_cleaned['voltage V'], *popt), label=f'Sigmoid Fit ({color.upper()})', color=color)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (pA)')
    plt.legend()
    plt.show()

# Find th