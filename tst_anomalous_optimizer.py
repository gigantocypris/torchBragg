import matplotlib.pyplot as plt
import torch
from exafel_project.kpp_utils.phil import parse_input
from tst_sf_linearity import get_wavelengths, get_fp_fdp, get_base_structure_factors, construct_structure_factors, get_Fhkl_mat
from amplitudes_spread_torchBragg import amplitudes_spread_psii
from tst_anomalous_optimizer_helper import set_all_params, forward_sim
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# Define ground truth fp and fdp for each of the 4 Mn atoms and calculate base structure factors
params,options = parse_input()

hkl_ranges=(11, -11, 22, -22, 30, -30)
direct_algo_res_limit=10.0
num_pixels = 128

# hkl_ranges=(1, -55, 6, -92, 134, -77)
# direct_algo_res_limit=1.85
# num_pixels = 3840


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
add_spots = True
use_background = True


# Forward simulation with the fp and fdp of the 4 Mn atoms set to the ground truth values ("experimental" data)
Fhkl_mat_vec = construct_structure_factors(fp_vec, fdp_vec, Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff) # shape is (num_wavelengths, 2*h_max+1, 2*k_max+1, 2*l_max+1)
Fhkl_mat_vec = torch.abs(Fhkl_mat_vec)
Fhkl_mat_vec = Fhkl_mat_vec.to(device)
all_params = set_all_params(params, sfall_channels, device)
experimental_data = forward_sim(params, all_params, Fhkl_mat_vec, add_spots, 
                use_background, hkl_ranges, device, num_pixels)

plt.figure(); plt.imshow(experimental_data.cpu().detach().numpy(), cmap='Greys',vmax=500); plt.colorbar(); plt.savefig('experimental_data.png')

plt.figure()
plt.imshow(experimental_data.cpu().detach().numpy(), cmap='Greys', vmax=5.0e2)
plt.title('Experimental data')
plt.colorbar()
plt.savefig('experimental_data.png')

breakpoint()
# Initialize fp_guess and fdp_guess for each of the 4 Mn atoms
# Initialize to ground state Mn
fp_vec_ground_state, fdp_vec_ground_state = get_fp_fdp(wavelengths, num_wavelengths,
                                                       ["Mn_ground_state","Mn_ground_state","Mn_ground_state","Mn_ground_state"])
                                                    #    ["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"])
fp_guess = fp_vec_ground_state.clone().detach().to(device).requires_grad_(True) # shape is (num_Mn_atoms, num_wavelengths)
fdp_guess = fdp_vec_ground_state.clone().detach().to(device).requires_grad_(True) # shape is (num_Mn_atoms, num_wavelengths)

Fhkl_mat_0 = Fhkl_mat_0.to(device)
Fhkl_mat_vec_all_Mn_diff = Fhkl_mat_vec_all_Mn_diff.to(device)
optimizer = torch.optim.Adam([fp_guess, fdp_guess], lr=0.1) 


# Define the loss function as the distance between the "experimental" data and the forward simulation result
# Either MSE or negative log-likelihood
num_iter = 10
loss_vec = []
for i in range(num_iter):
    # Zero out gradients
    optimizer.zero_grad()
    # Forward simulation with the saved parameters and fp_guess, fdp_guess, resulting in probability distribution
    Fhkl_mat_vec_guess = construct_structure_factors(fp_guess, fdp_guess, Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff) # shape is (num_wavelengths, 2*h_max+1, 2*k_max+1, 2*l_max+1)
    Fhkl_mat_vec_guess_amplitude = torch.abs(Fhkl_mat_vec_guess).to(device)
    simulated_data = forward_sim(params, all_params, Fhkl_mat_vec_guess_amplitude, add_spots, 
                                 use_background, hkl_ranges, device, num_pixels) # XXX modify to output a distribution
    # Calculate loss
    loss = torch.sum((simulated_data - experimental_data)**2)
    # Print loss
    print(loss)
    loss_vec.append(loss.detach().cpu().numpy())
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

plt.figure()
plt.plot(loss_vec)
plt.title('Loss')
plt.savefig('loss.png')

for mn in range(len(MN_labels)):
    plt.figure()
    plt.plot(torch.squeeze(fp_vec[mn]).cpu().detach().numpy(), '.', label='ground truth')
    plt.plot(torch.squeeze(fp_guess[mn]).cpu().detach().numpy(), '.', label='predicted')
    plt.legend()
    plt.title('fp ' + str(mn))
    plt.savefig('fp_optimized' + str(mn) + '.png')

    plt.figure()
    plt.plot(torch.squeeze(fdp_vec[mn]).cpu().detach().numpy(), '.', label='ground truth')
    plt.plot(torch.squeeze(fdp_guess[mn]).cpu().detach().numpy(), '.', label='predicted')
    plt.legend()
    plt.title('fdp')
    plt.savefig('fdp_optimized' + str(mn) + '.png')

# figure with experimental and simulated data side by side as subplots
vmin = experimental_data.cpu().detach().numpy().min()
vmax = 5.0e2 # experimental_data.cpu().detach().numpy().max()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(experimental_data.cpu().detach().numpy(), cmap='Greys', vmin=vmin, vmax=vmax)
plt.title('Experimental data')
plt.colorbar()


plt.subplot(1,2,2)
plt.imshow(simulated_data.cpu().detach().numpy(), cmap='Greys', vmin=vmin, vmax=vmax)
plt.title('Simulated data')
plt.colorbar()
plt.savefig('experimental_simulated.png')

breakpoint()