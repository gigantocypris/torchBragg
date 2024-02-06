import torch
from exafel_project.kpp_utils.phil import parse_input
from tst_sf_linearity import get_wavelengths, get_fp_fdp, get_base_structure_factors, construct_structure_factors, get_Fhkl_mat
from tst_torchBragg_psii import amplitudes_spread_psii, set_basic_params, tst_one_CPU, tst_one_pytorch
torch.autograd.set_detect_anomaly(True)

# Define ground truth fp and fdp for each of the 4 Mn atoms and calculate base structure factors
params,options = parse_input()

hkl_ranges=(11, -11, 22, -22, 30, -30)
direct_algo_res_limit=2.1
MN_labels=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"]

wavelengths, num_wavelengths = get_wavelengths(params)
fp_vec, fdp_vec = get_fp_fdp(wavelengths,num_wavelengths, MN_labels)
Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff = get_base_structure_factors(params, direct_algo_res_limit=direct_algo_res_limit, hkl_ranges=hkl_ranges, MN_labels=MN_labels)


# Get parameters for the forward simulation with nanoBragg # XXX refactor this code to remove nanoBragg dependency
sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit)
basic_params = set_basic_params(params, sfall_channels, direct_algo_res_limit)
num_wavelengths = params.spectrum.nchannels
add_spots = True
use_background = True
num_pixels = 128
raw_pixels, nanoBragg_params, noise_params, fluence_background = tst_one_CPU(params, basic_params, sfall_channels, add_spots, use_background, 
                                                                             direct_algo_res_limit=direct_algo_res_limit, num_pixels=num_pixels)    

breakpoint()

nanoBragg_params = (SIM.phisteps, SIM.pix0_vector_mm, SIM.fdet_vector, SIM.sdet_vector, SIM.odet_vector, 
                        SIM.beam_vector, SIM.polar_vector, SIM.close_distance_mm, fluence_vec, 
                        SIM.beam_center_mm, SIM.spot_scale, SIM.curved_detector, SIM.point_pixel,
                        SIM.integral_form, SIM.nopolar)

noise_params = (SIM.quantum_gain, SIM.detector_calibration_noise_pct, SIM.flicker_noise_pct, SIM.readout_noise_adu, \
        SIM.detector_psf_type, SIM.detector_psf_fwhm_mm, SIM.detector_psf_kernel_radius_pixels)

noise_params = (quantum_gain, detector_calibration_noise_pct, flicker_noise_pct, readout_noise_adu, \
        detector_psf_type, detector_psf_fwhm_mm, detector_psf_kernel_radius_pixels)

fluence_background

nanoBragg_params = phisteps, pix0_vector_mm, fdet_vector, sdet_vector, odet_vector, beam_vector, polar_vector, \
    close_distance_mm, fluence_vec, beam_center_mm, spot_scale, curved_detector, point_pixel, \
    integral_form, nopolar

# Forward simulation with the fp and fdp of the 4 Mn atoms set to the ground truth values ("experimental" data)
Fhkl_mat_vec = construct_structure_factors(fp_vec, fdp_vec, Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff) # shape is (num_wavelengths, 2*h_max+1, 2*k_max+1, 2*l_max+1)
Fhkl_mat_vec = torch.abs(Fhkl_mat_vec)
breakpoint()
experimental_data = tst_one_pytorch(params, basic_params, Fhkl_mat_vec, add_spots, nanoBragg_params, noise_params, fluence_background, use_background, hkl_ranges, 
                                    direct_algo_res_limit=direct_algo_res_limit, num_pixels=num_pixels)


# Initialize fp_guess and fdp_guess for each of the 4 Mn atoms
# XXX initialize to ground state Mn
fp_guess = torch.zeros_like(fp_vec, requires_grad=True).to(device)
fdp_guess = torch.zeros_like(fdp_vec, requires_grad=True).to(device)

optimizer = torch.optim.Adam([fp_guess, fdp_guess], lr=0.1)

# Define the loss function as the distance between the "experimental" data and the forward simulation result
# Either MSE or negative log-likelihood
num_iter = 10
for i in range(num_iter):
    # Zero out gradients
    optimizer.zero_grad()
    # Forward simulation with the saved parameters and fp_guess, fdp_guess, resulting in probability distribution
    Fhkl_mat_vec_guess = construct_structure_factors(fp_guess, fdp_guess, Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff) # shape is (num_wavelengths, 2*h_max+1, 2*k_max+1, 2*l_max+1)
    Fhkl_mat_vec_guess_amplitude = torch.abs(Fhkl_mat_vec_guess)
    
    simulated_data = tst_one_pytorch(params, basic_params, Fhkl_mat_vec_guess_amplitude, add_spots, nanoBragg_params, noise_params, fluence_background, use_background, hkl_ranges, 
                                     direct_algo_res_limit=direct_algo_res_limit, num_pixels=num_pixels) # XXX modify to output a distribution
    # Calculate loss
    loss = torch.sum((simulated_data - experimental_data)**2)
    # Print loss
    print(loss)
    # Backpropagate loss
    loss.backward()
    # Update fp_guess and fdp_guess
    optimizer.step()

print("fp_vec:")
print(torch.squeeze(fp_vec))
print("fp_guess:")
print(torch.squeeze(fp_guess))

print("fdp_vec:")
print(torch.squeeze(fdp_vec))
print("fdp_guess:")
print(torch.squeeze(fdp_guess))


breakpoint()