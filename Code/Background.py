import numpy as np
import matplotlib.pyplot as plt

# Load thebackgorund data 

voltage, current, unc = np.loadtxt('Data/Background.csv', delimiter=',', skiprows=1, unpack=True)

# Plot the background data

plt.errorbar(voltage, current, yerr=unc, fmt='o')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (pA)')
plt.title('Background data')
plt.show()
