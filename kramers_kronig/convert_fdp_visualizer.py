import matplotlib.pyplot as plt

def create_figures_full_cubic_spline(energy_vec_reference, fp_vec_reference, fdp_vec_reference, energy_vec_physical, fp_vec_physical, fdp_vec_physical, bandedge, prefix="Mn"):
    
    # reference curves
    plt.figure(figsize=(20,10))
    plt.figure()
    plt.plot(energy_vec_reference, fp_vec_reference, 'r', label="fp")
    plt.plot(energy_vec_reference, fdp_vec_reference, 'b', label="fdp")
    plt.legend()
    plt.savefig(prefix + "_fp_fdp_reference.png")

    # fdp
    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_reference, fdp_vec_reference, 'r', label="fdp reference")
    plt.plot(energy_vec_physical, fdp_vec_physical, 'b', label="fdp calculated")
    plt.legend()
    plt.savefig(prefix + "_fdp.png")

    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_reference, fdp_vec_reference, 'r', label="fdp reference")
    plt.plot(energy_vec_physical, fdp_vec_physical, 'b', label="fdp calculated")
    plt.legend()
    plt.xlim([bandedge - 50, bandedge + 50])
    plt.savefig(prefix + "_fdp_bandwidth.png")


    # fp
    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_reference, fp_vec_reference, 'r', label="fp reference")
    plt.plot(energy_vec_physical, fp_vec_physical, 'b', label="fp calculated")
    plt.legend()
    plt.savefig(prefix + "_fp.png")

    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_reference, fp_vec_reference, 'r', label="fp reference")
    plt.plot(energy_vec_physical, fp_vec_physical, 'b', label="fp calculated")
    plt.legend()
    plt.xlim([bandedge - 50, bandedge + 50])
    plt.savefig(prefix + "_fp_bandwidth.png")

    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_physical, fdp_vec_physical, 'r', label="fdp calculated")
    plt.plot(energy_vec_physical, fp_vec_physical, 'b', label="fp calculated")
    plt.legend()
    plt.xlim([bandedge - 50, bandedge + 50])
    plt.savefig(prefix + "_fdp_fp_bandwidth.png")
    
def create_figures(energy_vec_reference, energy_vec_bandwidth, fp_vec_reference, fdp_vec_reference, energy_vec_final, fdp_final, fp_final, fdp_check, prefix="Mn"):
    
    # reference curves
    plt.figure(figsize=(20,10))
    plt.figure()
    plt.plot(energy_vec_reference, fp_vec_reference, 'r', label="fp")
    plt.plot(energy_vec_reference, fdp_vec_reference, 'b', label="fdp")
    plt.legend()
    plt.savefig(prefix + "_fp_fdp_reference.png")

    # fdp
    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_reference, fdp_vec_reference, 'r', label="fdp reference")
    plt.plot(energy_vec_final, fdp_final, 'b', label="fdp calculated")
    plt.legend()
    plt.savefig(prefix + "_fdp.png")

    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_reference, fdp_vec_reference, 'r', label="fdp reference")
    plt.plot(energy_vec_final, fdp_final, 'b', label="fdp calculated")
    plt.legend()
    plt.xlim([energy_vec_bandwidth[0],energy_vec_bandwidth[-1]])
    plt.savefig(prefix + "_fdp_bandwidth.png")

    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_final, fdp_final, 'r', label="fdp calc")
    plt.plot(energy_vec_final, fdp_check, 'b', label="fdp calc reformatted coeff")
    plt.legend()
    plt.xlim([energy_vec_bandwidth[0],energy_vec_bandwidth[-1]])
    plt.savefig(prefix + "_fdp_check_bandwidth.png")

    # fp
    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_reference, fp_vec_reference, 'r', label="fp reference")
    plt.plot(energy_vec_final, fp_final, 'b', label="fp calculated")
    plt.legend()
    plt.savefig(prefix + "_fp.png")

    plt.figure(figsize=(20,10))
    plt.plot(energy_vec_reference, fp_vec_reference, 'r', label="fp reference")
    plt.plot(energy_vec_final, fp_final, 'b', label="fp calculated")
    plt.legend()
    plt.xlim([energy_vec_bandwidth[0], energy_vec_bandwidth[-1]])
    plt.savefig(prefix + "_fp_bandwidth.png")

