import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Sample data
num_points = 10
x = np.arange(0,num_points+1,1)
x1= np.arange(0,num_points+0.1,0.1)
y = np.random.rand(num_points+1)

# Create a CubicSpline
cs = CubicSpline(x, y)

# Extract coefficients
coefficients = cs.c

# Print coefficients
print("Coefficients:", coefficients)
breakpoint()
plt.figure()
plt.plot(x, y, 'r*', label="data")
plt.plot(x1, cs(x1), label="cs")
plt.legend()
plt.savefig("cs.png")

