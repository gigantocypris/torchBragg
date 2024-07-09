import matplotlib.pyplot as plt

def create_figures(energy_vec_reference, energy_vec_bandwidth, fp_vec_reference, fdp_vec_reference, energy_vec_final, fdp_final, fp_final, prefix="Mn"):
    
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

