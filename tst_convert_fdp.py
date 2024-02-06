import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from create_fp_fdp_dat_file import full_path, read_dat_file
from tst_convert_fdp_helper import func, convert_coeff, func_converted_coeff, find_fdp, plot_fit, fdp_fp_integrate
np.seterr(all='raise')

# Path to Sherrell data: $MODULES/ls49_big_data/data_sherrell
# Mn_model=full_path("data_sherrell/Mn.dat") # Mn, MnO2_spliced
# Mn_model=full_path("data_sherrell/Fe.dat")
Mn_model=full_path("data_sherrell/Mn.dat")
relativistic_correction = 0 # 0.042 for Mn and 0.048 for Fe
bandedge = 6550 # 6550 eV is the bandedge of Mn and 7112 is the bandedge of Fe


energy_vec, fp_vec, fdp_vec = read_dat_file(Mn_model)
energy_vec = np.array(energy_vec).astype(np.float64)
fp_vec = np.array(fp_vec).astype(np.float64)
fdp_vec = np.array(fdp_vec).astype(np.float64)

# For Mn, split curve from start to 6499.5eV, from 6499.5eV to 6599.5, 6599.5eV to end 
# 6550 eV is the bandedge of Mn
interval_0 = np.array([energy_vec[0], bandedge - 50.5])
interval_1 = np.array([bandedge - 50.5, bandedge + 49.5]) 
interval_2 = np.array([bandedge + 49.5, energy_vec[-1]])


plt.figure(figsize=(20,10))
plt.plot(energy_vec, fp_vec, label="fp")
plt.plot(energy_vec, fdp_vec, label="fdp")
plt.legend()
plt.savefig("Mn_fp_fdp.png")

cs_fdp = CubicSpline(energy_vec, fdp_vec)

plt.figure(figsize=(20,10))
plt.plot(energy_vec, fdp_vec, 'r.', label="fdp")
plt.plot(energy_vec, cs_fdp(energy_vec), label="cs_fdp")
plt.legend()
plt.savefig("Mn_fdp_cs.png")




# Fit interval_1 with a cubic spline
step_size = 1
energy_vec_bandwidth = np.arange(interval_1[0], interval_1[1] + step_size, step_size)
fdp_vec_bandwidth = cs_fdp(energy_vec_bandwidth)

cs_fdp_bandwidth = CubicSpline(energy_vec_bandwidth, fdp_vec_bandwidth)
coeff_bandwidth = cs_fdp_bandwidth.c

plt.figure()
plt.plot(energy_vec_bandwidth, fdp_vec_bandwidth, 'b.', label="fdp")
plt.plot(energy_vec_bandwidth, cs_fdp_bandwidth(energy_vec_bandwidth), 'r', label="cs_fdp")
plt.legend()
plt.savefig("Mn_fdp_cs_bandwidth.png")

# Fit interval_0 with a polynomial, enforce endpoint continuity

energy_vec_0 = energy_vec[energy_vec <= interval_0[1]]
fdp_vec_0 = fdp_vec[energy_vec <= interval_0[1]]
if energy_vec_0[-1] != interval_0[1]:
    energy_vec_0 = np.append(energy_vec_0, interval_0[1])
    fdp_vec_0 = np.append(fdp_vec_0, cs_fdp_bandwidth(interval_0[1]))

shift_0 = energy_vec_0[-1]
constant_0 = fdp_vec_0[-1]

func_fixed_pt_0 = lambda x, a, b, c, d, e: func(x, shift_0, constant_0, a, b, c, d, e)

popt_0, pcov_0 = curve_fit(func_fixed_pt_0, energy_vec_0, fdp_vec_0)   

plt.figure()
plt.plot(energy_vec_0, fdp_vec_0, 'b.', label="fdp")
plt.plot(energy_vec_0, func_fixed_pt_0(energy_vec_0, *popt_0), 'r', label="fit")
plt.legend()
plt.savefig("Mn_fdp_fit_0.png")

# Fit interval_1 with a polynomial, enforce endpoint continuity

energy_vec_2 = energy_vec[energy_vec >= interval_2[0]]
fdp_vec_2 = fdp_vec[energy_vec >= interval_2[0]]
if energy_vec_2[0] != interval_2[0]:
    energy_vec_2 = np.append(interval_2[0], energy_vec_2)
    fdp_vec_2 = np.append(cs_fdp_bandwidth(interval_2[0]), fdp_vec_2)

shift_2 = energy_vec_2[0]
constant_2 = fdp_vec_2[0]

func_fixed_pt_2 = lambda x, a, b, c, d, e: func(x, shift_2, constant_2, a, b, c, d, e)

popt_2, pcov_2 = curve_fit(func_fixed_pt_2, energy_vec_2, fdp_vec_2)   

plt.figure()
plt.plot(energy_vec_2, fdp_vec_2, 'b.', label="fdp")
plt.plot(energy_vec_2, func_fixed_pt_2(energy_vec_2, *popt_2), 'r', label="fit")
plt.legend()
plt.savefig("Mn_fdp_fit_2.png")

# Plot the entire fit

energy_vec_full = np.concatenate((energy_vec_0, energy_vec_bandwidth, energy_vec_2), axis=0)
fdp_vec_full = np.concatenate((func_fixed_pt_0(energy_vec_0, *popt_0), cs_fdp_bandwidth(energy_vec_bandwidth), func_fixed_pt_2(energy_vec_2, *popt_2)), axis=0)

plt.figure(figsize=(20,10))
plt.plot(energy_vec, fdp_vec, 'r.', label="original")
plt.plot(energy_vec_full, fdp_vec_full, 'b', label="fdp fit")
plt.legend()
plt.savefig("Mn_fdp_full.png")

# Create intervals_mat, coeff_mat, powers_mat

# interval_1 is broken up into intervals depending on step size
interval_1_starts = np.arange(interval_1[0], interval_1[1], step_size)
interval_1_ends = np.arange(interval_1[0]+step_size, interval_1[1] + step_size, step_size)
interval_1_all = np.array([interval_1_starts, interval_1_ends]).T

intervals_mat = np.concatenate([np.expand_dims(interval_0,axis=0), interval_1_all, np.expand_dims(interval_2, axis=0)],axis=0) # intervals x endpoints


powers_mat = np.array([-2,-1,0,1,2,3])
powers_mat = np.repeat(np.expand_dims(powers_mat, axis=0), len(intervals_mat), axis=0) # intervals x powers

coeff_vec_0 = np.expand_dims(convert_coeff(shift_0, constant_0, *popt_0), axis=0)


plt.figure()
plt.plot(energy_vec_0, fdp_vec_0, 'b.', label="fdp")
plt.plot(energy_vec_0, func_converted_coeff(energy_vec_0, np.array([-2,-1,0,1,2,3]), convert_coeff(shift_0, constant_0, *popt_0)), 'g', label="fit2")
plt.legend()
plt.savefig("Mn_fdp_fit2_0.png")


coeff_vec_1 = []
for i in range(len(interval_1_all)):
    coeff_vec_1.append(convert_coeff(interval_1_all[i,0], coeff_bandwidth[3,i], coeff_bandwidth[0,i], coeff_bandwidth[1,i], coeff_bandwidth[2,i], 0, 0))
coeff_vec_1 = np.stack(coeff_vec_1, axis=0)
coeff_vec_2 = np.expand_dims(convert_coeff(shift_2, constant_2, *popt_2), axis=0)

coeff_mat = np.concatenate([coeff_vec_0, coeff_vec_1, coeff_vec_2], axis=0)

plot_fit(powers_mat, coeff_mat, intervals_mat, energy_vec, fdp_vec)

# Now convert fdp to fp, account for relativistic correction

energy_vec_bandwidth = np.arange(bandedge-50, bandedge + 50, 1.)
# energy_vec_bandwidth = np.concatenate((np.arange(1100,bandedge-50,100.), np.arange(bandedge-50, bandedge + 50,1.), np.arange(bandedge + 50, 10000, 100.)))
fdp_calculate_bandwidth = []
fp_calculate_bandwidth = []

for energy in energy_vec_bandwidth:
    fdp_calculate_bandwidth.append(find_fdp(energy, powers_mat, coeff_mat, intervals_mat))
    fp = fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction)
    fp_calculate_bandwidth.append(fp)

plt.figure()
plt.plot(energy_vec, fp_vec, 'r.', label="original")
plt.plot(energy_vec_bandwidth, fp_calculate_bandwidth, 'b', label="calculated")
plt.xlim([energy_vec_bandwidth[0],energy_vec_bandwidth[-1]])
plt.legend()
plt.savefig("Mn_fp_calculated.png")
