import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Load the data for each color in a for loop
data_dict = {}

for i in ['Blue', 'Green', 'Red', 'Yellow']:
    data_dict[i] = pd.read_csv('Data/Hyperfine/{}.csv'.format(i))

print(data_dict)

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
    plt.errorbar(voltage, current, yerr=unc, fmt = 'x')
    plt.title(f'{color} data')
    
    plt.plot(voltage, shockley_diode(voltage, *popt))
    plt.show()

    print(f"The value of A, B, C for {color} data is: {popt}")
