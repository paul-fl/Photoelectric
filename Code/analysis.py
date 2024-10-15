import numpy as np
import matplotlib.pyplot as plt

v_1, a_1, unc_1 = np.loadtxt('data.txt', unpack=True)

# Plot the data

plt.errorbar(v_1, a_1, yerr=unc_1, fmt='o')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')



