import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 1, 2, 0, 1, 3, 2, 4, 3, 5])

# Define the intervals for cubic polynomial fitting
intervals = [(1, 4), (4, 7), (7, 10)]

# Create an empty array to store the coefficients of the cubic polynomials
coefficients = []

# Perform cubic polynomial fitting on each interval
for i in range(len(intervals) - 1):
    # Extract data within the current interval
    mask = (x >= intervals[i][0]) & (x <= intervals[i + 1][1])
    x_interval = x[mask]
    y_interval = y[mask]

    # Fit a cubic polynomial (degree=3) to the interval
    coeffs = np.polyfit(x_interval, y_interval, 3)
    
    # Append the coefficients to the list
    coefficients.extend(coeffs[:-1])  # Exclude the constant term

# Define the objective function to minimize
def objective_function(coeffs, x_vals, intervals):
    y_vals = np.polyval(np.append(coeffs, 0), x_vals)
    total_error = 0

    # Add error terms for continuity at interval boundaries
    for i in range(len(intervals) - 1):
        boundary_x = intervals[i + 1][0]
        y1 = np.polyval(np.append(coeffs[i * 3 : (i + 1) * 3], 0), boundary_x)
        y2 = np.polyval(np.append(coeffs[(i + 1) * 3 : (i + 2) * 3], 0), boundary_x)
        total_error += (y1 - y2)**2

    return total_error

# Initial guess for the coefficients
initial_coeffs = np.zeros(len(intervals) * 3 - 1)

# Minimize the objective function to find the coefficients
result = minimize(objective_function, initial_coeffs, args=(x, intervals), method='BFGS')

# Extract the optimized coefficients
optimized_coeffs = result.x

# Plot the original data points
plt.scatter(x, y, label='Data Points', color='blue')

# Plot the overall cubic polynomial function
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = np.polyval(np.append(optimized_coeffs, 0), x_vals)
plt.plot(x_vals, y_vals, label='Cubic Polynomial', color='red')

# Set labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Cubic Polynomial Fitting with Continuity')
plt.savefig("cubic_polynomial_fitting.png")
