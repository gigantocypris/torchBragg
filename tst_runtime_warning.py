import numpy as np
import warnings
np.seterr(all='raise')

def log_energy_ratio(energy_start, energy_end, energy):
    try:
        return np.log((energy_end - energy)/(energy_start - energy))
    except FloatingPointError as rw:
        print("FloatingPointError:", rw)
        breakpoint()

result = log_energy_ratio(1, 2, 2)  # This will raise a FloatingPointError
