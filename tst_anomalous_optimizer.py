import torch
from exafel_project.kpp_utils.phil import parse_input
from simtbx.nanoBragg import shapetype
from tst_sf_linearity import get_wavelengths, get_fp_fdp, get_base_structure_factors, construct_structure_factors, get_Fhkl_mat
from tst_torchBragg_psii import amplitudes_spread_psii, set_basic_params, tst_one_CPU, tst_one_pytorch
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ground truth fp and fdp for each of the 4 Mn atoms and calculate base structure factors
params,options = parse_input()

hkl_ranges=(11, -11, 22, -22, 30, -30)
direct_algo_res_limit=10.0
MN_labels=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"]

wavelengths, num_wavelengths = get_wavelengths(params)
fp_vec, fdp_vec = get_fp_fdp(wavelengths,num_wavelengths, MN_labels)

try:
    Fhkl_mat_0 = torch.load('Fhkl_mat_0.pt')
    Fhkl_mat_vec_all_Mn_diff = torch.load('Fhkl_mat_vec_all_Mn_diff.pt')
except FileNotFoundError:
    Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff = get_base_structure_factors(params, direct_algo_res_limit=direct_algo_res_limit, hkl_ranges=hkl_ranges, MN_labels=MN_labels)
    torch.save(Fhkl_mat_0, 'Fhkl_mat_0.pt')
    torch.save(Fhkl_mat_vec_all_Mn_diff, 'Fhkl_mat_vec_all_Mn_diff.pt')


# Get parameters for the forward simulation 
sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit)
basic_params = set_basic_params(params, sfall_channels)
add_spots = True
use_background = True
num_pixels = 128 

phisteps = 1
pix0_vector_mm = (141.7, 5.72, -5.72) # detector origin, change to get different ROI #XXX
fdet_vector = (0.0, 0.0, 1.0)
sdet_vector = (0.0, -1.0, 0.0)
odet_vector = (1.0, 0.0, 0.0)
beam_vector = (1.0, 0.0, 0.0)
polar_vector = (0.0, 0.0, 1.0)
close_distance_mm = 141.7
fluence_vec = [1.374132034195255e+19, 1.0686071593132872e+19]
beam_center_mm = (5.675999999999999, 5.675999999999999)
spot_scale = 1.0
curved_detector = False
point_pixel = False
integral_form = False
nopolar = False

nanoBragg_params = (phisteps, pix0_vector_mm, fdet_vector, sdet_vector, odet_vector, beam_vector, polar_vector, 
                    close_distance_mm, fluence_vec, beam_center_mm, spot_scale, curved_detector, point_pixel, 
                    integral_form, nopolar)

quantum_gain = 1.0
detector_calibration_noise_pct = 1.0
flicker_noise_pct = 0.0
readout_noise_adu = 1.0
detector_psf_type = shapetype.Unknown
detector_psf_fwhm_mm = 0.0
detector_psf_kernel_radius_pixels = 0

noise_params = (quantum_gain, detector_calibration_noise_pct, flicker_noise_pct, readout_noise_adu, 
                detector_psf_type, detector_psf_fwhm_mm, detector_psf_kernel_radius_pixels)

fluence_background = 1.1111111111111111e+23

# Forward simulation with the fp and fdp of the 4 Mn atoms set to the ground truth values ("experimental" data)
Fhkl_mat_vec = construct_structure_factors(fp_vec, fdp_vec, Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff) # shape is (num_wavelengths, 2*h_max+1, 2*k_max+1, 2*l_max+1)
Fhkl_mat_vec = torch.abs(Fhkl_mat_vec)
experimental_data = tst_one_pytorch(params, basic_params, Fhkl_mat_vec, add_spots, nanoBragg_params, noise_params, fluence_background, use_background, hkl_ranges, 
                                    num_pixels=num_pixels)


# Initialize fp_guess and fdp_guess for each of the 4 Mn atoms
# Initialize to ground state Mn

fp_vec_ground_state, fdp_vec_ground_state = get_fp_fdp(wavelengths, num_wavelengths, ["Mn_ground_state","Mn_ground_state","Mn_ground_state","Mn_ground_state"]) # ground state

fp_guess = fp_vec_ground_state.clone().detach().to(device).requires_grad_(True) # shape is (num_Mn_atoms, num_wavelengths)
fdp_guess = fdp_vec_ground_state.clone().detach().to(device).requires_grad_(True) # shape is (num_Mn_atoms, num_wavelengths)

Fhkl_mat_0 = Fhkl_mat_0.to(device)
Fhkl_mat_vec_all_Mn_diff = Fhkl_mat_vec_all_Mn_diff.to(device)
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
                                     num_pixels=num_pixels) # XXX modify to output a distribution
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