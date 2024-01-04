import os
import math
import matplotlib.pyplot as plt
from exafel_project.kpp_utils.phil import parse_input
from LS49.spectra.generate_spectra import spectra_simulation
from LS49.sim.step4_pad import microcrystal
from LS49 import ls49_big_data, legacy_random_orientations
from exafel_project.kpp_utils.ferredoxin import basic_detector_rayonix
from torchBragg.amplitudes_spread_torchBragg import amplitudes_spread_psii
from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg import shapetype
from scitbx.array_family import flex
import scitbx
from scitbx.matrix import sqr,col

def tst_one_CPU(params, use_background):
    spectra = spectra_simulation()
    crystal = microcrystal(Deff_A = params.crystal.Deff_A, length_um = params.crystal.length_um, beam_diameter_um = 1.0) # assume smaller than 10 um crystals
    # random_orientation = legacy_random_orientations(100)[0]
    random_orientation = legacy_random_orientations(1)[0]
    
    DETECTOR = basic_detector_rayonix()
    PANEL = DETECTOR[0]
    sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=10)

    iterator = spectra.generate_recast_renormalized_image_parameterized(image=0,params=params)
    rand_ori = sqr(random_orientation)

    wavlen, flux, shot_to_shot_wavelength_A = next(iterator) # list of lambdas, list of fluxes, average wavelength

    assert shot_to_shot_wavelength_A > 0 # wavelength varies shot-to-shot

    # use crystal structure to initialize Fhkl array
    N = crystal.number_of_cells(sfall_channels[0].unit_cell())

    detpixels_slowfast=PANEL.get_image_size()
    # detpixels_slowfast=(512,512)

    SIM = nanoBragg(detpixels_slowfast=detpixels_slowfast,pixel_size_mm=PANEL.get_pixel_size()[0],Ncells_abc=(N,N,N),
                    wavelength_A=shot_to_shot_wavelength_A,verbose=0)
    SIM.adc_offset_adu = 0 # Do not offset by 40
    SIM.mosaic_spread_deg = 0.05 # interpreted by UMAT_nm as a half-width stddev
                                # mosaic_domains setter MUST come after mosaic_spread_deg setter
    SIM.mosaic_domains = int(os.environ.get("MOS_DOM"))
    print ("MOSAIC",SIM.mosaic_domains)
    SIM.distance_mm = PANEL.get_distance()

    UMAT_nm = flex.mat3_double()
    mersenne_twister = flex.mersenne_twister(seed=0)
    scitbx.random.set_random_seed(1234)
    rand_norm = scitbx.random.normal_distribution(mean=0, sigma=SIM.mosaic_spread_deg * math.pi/180.)
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(SIM.mosaic_domains)
    for m in mosaic_rotation:
        site = col(mersenne_twister.random_double_point_on_sphere())
        UMAT_nm.append( site.axis_and_angle_as_r3_rotation_matrix(m,deg=False) )
    SIM.set_mosaic_blocks(UMAT_nm)

    if params.attenuation:
        SIM.detector_thick_mm = 0.032 # = 0 for Rayonix
        SIM.detector_thicksteps = 1 # should default to 1 for Rayonix, but set to 5 for CSPAD
        SIM.detector_attenuation_length_mm = 0.017 # default is silicon

    # get same noise each time this test is run
    SIM.seed = 1
    SIM.oversample=1
    SIM.wavelength_A = shot_to_shot_wavelength_A
    SIM.polarization=1
    # this will become F000, marking the beam center
    SIM.default_F=0
    SIM.Fhkl=sfall_channels[0] # instead of sfall_main

    Amatrix_rot = (rand_ori*sqr(sfall_channels[0].unit_cell().orthogonalization_matrix())).transpose()

    SIM.Amatrix_RUB = Amatrix_rot
    print("SIM.Amatrix.RUB", SIM.Amatrix_RUB)
    #workaround for failing init_cell, use custom written Amatrix setter
    print("unit_cell_Adeg=",SIM.unit_cell_Adeg)
    print("unit_cell_tuple=",SIM.unit_cell_tuple)
    Amat = sqr(SIM.Amatrix).transpose() # recovered Amatrix from SIM
    # fastest option, least realistic
    SIM.xtal_shape=shapetype.Gauss_argchk # both crystal & RLP are Gaussian
    # only really useful for long runs
    SIM.progress_meter=False
    # prints out value of one pixel only.  will not render full image!
    # flux is always in photons/s
    SIM.flux=params.beam.total_flux
    SIM.exposure_s=1.0 # so total fluence is e12
    # assumes round beam
    SIM.beamsize_mm=0.003 #cannot make this 3 microns; spots are too intense
    temp=SIM.Ncells_abc
    SIM.Ncells_abc=temp

    # rough approximation to water: interpolation points for sin(theta/lambda) vs structure factor
    water_bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.18,7.32),(0.2,6.75),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
    assert [a[0] for a in water_bg] == sorted([a[0] for a in water_bg])
    # rough approximation to air
    air_bg = flex.vec2_double([(0,14.1),(0.045,13.5),(0.174,8.35),(0.35,4.78),(0.5,4.22)])
    assert [a[0] for a in air_bg] == sorted([a[0] for a in air_bg])


    # loop over energies
    for x in range(len(flux)):
        print("SIM.Amatrix.RUB", SIM.Amatrix_RUB)
        print("Wavelength",x)
        # from channel_pixels function
        SIM.wavelength_A = wavlen[x]
        SIM.flux = flux[x]
        SIM.Fhkl=sfall_channels[x]
        SIM.Amatrix_RUB = Amatrix_rot
        print("SIM.Amatrix.RUB", SIM.Amatrix_RUB)
        SIM.add_nanoBragg_spots()

    # simulated crystal is only 125 unit cells (25 nm wide)
    # amplify spot signal to simulate physical crystal of 4000x larger: 100 um (64e9 x the volume)
    SIM.raw_pixels *= crystal.domains_per_crystal # must calculate the correct scale!

    SIM.wavelength_A = shot_to_shot_wavelength_A # return to canonical energy for subsequent background
    SIM.Amatrix_RUB = Amatrix_rot # return to canonical orientation
    print("SIM.Amatrix.RUB", SIM.Amatrix_RUB)


    if use_background:
        SIM.Fbg_vs_stol = water_bg
        SIM.amorphous_sample_thick_mm = 0.1
        SIM.amorphous_density_gcm3 = 1
        SIM.amorphous_molecular_weight_Da = 18
        SIM.flux=params.beam.total_flux
        SIM.beamsize_mm=0.003 # square (not user specified)
        SIM.exposure_s=1.0 # multiplies flux x exposure
        SIM.add_background()
        SIM.Fbg_vs_stol = air_bg
        SIM.amorphous_sample_thick_mm = 10 # between beamstop and collimator
        SIM.amorphous_density_gcm3 = 1.2e-3
        SIM.amorphous_sample_molecular_weight_Da = 28 # nitrogen = N2
        SIM.add_background()

    if params.psf:
        SIM.detector_psf_kernel_radius_pixels=10
        SIM.detector_psf_type=shapetype.Fiber # for Rayonix
        SIM.detector_psf_fwhm_mm=0.08
        #SIM.apply_psf() # the actual application is called within the C++ SIM.add_noise()
    else:
        #SIM.detector_psf_kernel_radius_pixels=5;
        SIM.detector_psf_type=shapetype.Unknown # for CSPAD
        SIM.detector_psf_fwhm_mm=0

    if params.noise:
        SIM.detector_calibration_noise_pct = 1.0
        SIM.readout_noise_adu = 1.

    if params.noise or params.psf:
        print("quantum_gain=",SIM.quantum_gain) #defaults to 1. converts photons to ADU
        print("adc_offset_adu=",SIM.adc_offset_adu)
        print("detector_calibration_noise_pct=",SIM.detector_calibration_noise_pct)
        print("flicker_noise_pct=",SIM.flicker_noise_pct)
        print("readout_noise_adu=",SIM.readout_noise_adu) # gaussian random number to add to every pixel (0 for PAD)
        # apply Poissonion correction, then scale to ADU, then adc_offset.
        # should be 10 for most Rayonix, Pilatus should be 0, CSPAD should be 0.
        print("detector_psf_type=",SIM.detector_psf_type)
        print("detector_psf_fwhm_mm=",SIM.detector_psf_fwhm_mm)
        print("detector_psf_kernel_radius_pixels=",SIM.detector_psf_kernel_radius_pixels)
        SIM.add_noise() #converts photons to ADU.

    return SIM.raw_pixels



if __name__ == "__main__":
    params,options = parse_input()
    use_background = True
    raw_pixels = tst_one_CPU(params, use_background)
    if use_background:
        plt.figure(); plt.imshow(raw_pixels.as_numpy_array(), vmax=1e3);plt.savefig("raw_pixels_0.png")
    else:
        plt.figure(); plt.imshow(raw_pixels.as_numpy_array(), vmax=10e-5);plt.savefig("raw_pixels_0.png")
    
