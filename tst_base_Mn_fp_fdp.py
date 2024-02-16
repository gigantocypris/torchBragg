from __future__ import print_function, division
import os
from cctbx.dispersion.kramers_kronig import kramers_kronig 
from cctbx.dispersion.kramers_kronig import kramers_kronig_helper
import matplotlib.pyplot as plt
import numpy as np
import torch

"""
THIS FILE IS CANDIDATE FOR DELETION
"""

MODULES = os.getenv('MODULES')
data_path = MODULES + "/psii_spread/merging/application/annulus"

'''
Files MnO2.dat and Mn2O3.dat created by Daniel Tchon:

These files were spliced from two sources. First, both the real (f'')
and imaginary (f') parts of the anomalous scattering were downloaded from
the University of Washington X-ray Absorption Edges
[website](http://skuld.bmsc.washington.edu/scatter/AS_periodic.html).
In order to increase the resolution around the absorption edge area,
experimental Mn2O3 and MnO2 curves of f'' were received from a correspondence
with Vittal Yachandra. They data back to an article about X-ray damage to the
Mn4Ca complex in PSII ([link](https://doi.org/10.1073/pnas.0505207102)).

The Washington files are expressed in absolute units, whether the curves in
Yachandra files are unitless along the "y" axis. In order to match the latter
to the former, the f'' values in Yachandra files were offset and rescaled,
until a satisfactory visual agreement was found. The Mn2O3 data was offset by
+0.45 units and rescaled by a factor of +3.55. Likewise, the MnO2 data
was offset by +0.45 and rescaled by a factor of 3.31. The data points
within the range of energy covered by the Yachandra files were then removed
from the Washington files and replaced, thus splicing the f'' information.
This process was performed by Daniel Tchoń in March 2023.

Resulting files featured f'' curves which were both complete and accurate
around the edge, but now lacked f' component in the "Yachandra range".

In order to reconstruct the f' component, the Kramers-Kronig dispersion
is used in this script. We note that changing part of the f'' curve changes the entire
f' curve (the change is predominately in the same region change by the f'' curve).
'''

sf_MnO2 = kramers_kronig_helper.parse_data(data_path + "/MnO2.dat") 
sf_Mn2O3 = kramers_kronig_helper.parse_data(data_path + "/Mn2O3.dat")
sf_Mn = kramers_kronig_helper.parse_data(data_path + "/Mn.dat")



plt.figure()
plt.title("MnO2, fp")
plt.plot(sf_MnO2[:,0],sf_MnO2[:,1])
plt.savefig("MnO2_fp.png")

plt.figure()
plt.title("MnO2, fdp")
plt.plot(sf_MnO2[:,0],sf_MnO2[:,2])
# plt.xlim(6550-100, 6550+100)
plt.savefig("MnO2_fdp.png")


plt.figure()
plt.title("Mn2O3, fp")
plt.plot(sf_Mn2O3[:,0],sf_Mn2O3[:,1])
plt.savefig("Mn2O3_fp.png")

plt.figure()
plt.title("Mn2O3, fdp")
plt.plot(sf_Mn2O3[:,0],sf_Mn2O3[:,2])
# plt.xlim(6550-100, 6550+100)
plt.savefig("Mn2O3_fdp.png")

plt.figure()
plt.title("Mn, fp")
plt.plot(sf_Mn[:,0],sf_Mn[:,1])
plt.savefig("Mn_fp.png")

plt.figure()
plt.title("Mn, fdp")
plt.plot(sf_Mn[:,0],sf_Mn[:,2])
plt.savefig("Mn_fdp.png")


energy_0 = sf_MnO2[:,0]
f_p = sf_MnO2[:,1]
f_dp = sf_MnO2[:,2]

breakpoint()





# find break in f_p
break_f_p = np.isnan(f_p)
break_start = np.where(break_f_p)[0][0]
break_end = np.where(break_f_p)[0][-1]


plt.figure()
plt.title("MnO2, fdp")
plt.plot(sf_MnO2[:,0],sf_MnO2[:,1])
plt.plot(energy_interp_0,f_p_pred)
plt.savefig("MnO2.png")

interpolated_break_start = np.where(energy_interp_0>energy_0[break_start])[0][0]
interpolated_break_end = np.where(energy_interp_0<energy_0[break_end])[0][-1]

# interpolate f_p excluding break
f_p = kramers_kronig_helper.interpolate_torch(torch.tensor(energy_0[~break_f_p]), torch.tensor(f_p[~break_f_p]), energy_interp_0)

# merge f_p and f_p_pred
energy_overlap_pts = 100

f_p_pred_break = f_p_pred[interpolated_break_start:interpolated_break_end]
f_p_pred_overlap_0 = f_p_pred[interpolated_break_start-energy_overlap_pts:interpolated_break_start]
f_p_pred_overlap_1 = f_p_pred[interpolated_break_end:interpolated_break_end+energy_overlap_pts]

f_p_overlap_0 = f_p[interpolated_break_start-energy_overlap_pts:interpolated_break_start]
f_p_overlap_1 = f_p[interpolated_break_end:interpolated_break_end+energy_overlap_pts]

window = np.arange(energy_overlap_pts)/energy_overlap_pts
f_p_overlap_0 = f_p_overlap_0*(1-window) + f_p_pred_overlap_0*window
f_p_overap_1 = f_p_overlap_1*(window) + f_p_pred_overlap_1*(1-window)

f_p[interpolated_break_start:interpolated_break_end] = f_p_pred_break
f_p[interpolated_break_start-energy_overlap_pts:interpolated_break_start] = f_p_overlap_0
f_p[interpolated_break_end:interpolated_break_end+energy_overlap_pts] = f_p_overlap_1

plt.figure()
plt.title("MnO2, fdp reconstructed")
plt.plot(sf_MnO2[:,0][~break_f_p],sf_MnO2[:,1][~break_f_p])
plt.plot(energy_interp_0, f_p)
plt.savefig("MnO2_fdp.png")



