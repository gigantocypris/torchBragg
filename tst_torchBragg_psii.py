import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
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
from diffraction_vectorized import add_torchBragg_spots
from add_background_vectorized import add_background
from utils_vectorized import Fhkl_remove, Fhkl_dict_to_mat
from add_noise import add_noise
from tst_sf_linearity import get_Fhkl_mat
torch.set_default_dtype(torch.float64)


def set_basic_params(params, sfall_channels):
    spectra = spectra_simulation()
    crystal = microcrystal(Deff_A = params.crystal.Deff_A, length_um = params.crystal.length_um, beam_diameter_um = 1.0) # assume smaller than 10 um crystals
    # random_orientation = legacy_random_orientations(100)[0]
    random_orientation = legacy_random_orientations(1)[0]
    
    DETECTOR = basic_detector_rayonix()
    PANEL = DETECTOR[0]
    
    iterator = spectra.generate_recast_renormalized_image_parameterized(image=0,params=params)
    rand_ori = sqr(random_orientation)

    wavlen, flux, shot_to_shot_wavelength_A = next(iterator) # list of lambdas, list of fluxes, average wavelength

    assert shot_to_shot_wavelength_A > 0 # wavelength varies shot-to-shot

    # use crystal structure to initialize Fhkl array
    N = crystal.number_of_cells(sfall_channels[0].unit_cell())
    
    pixel_size_mm=PANEL.get_pixel_size()[0]
    Ncells_abc=(N,N,N)

    adc_offset_adu = 0 # Do not offset by 40
    mosaic_spread_deg = 0.05 # interpreted by UMAT_nm as a half-width stddev
                                # mosaic_domains setter MUST come after mosaic_spread_deg setter
    mosaic_domains = int(os.environ.get("MOS_DOM"))
    distance_mm = PANEL.get_distance()

    UMAT_nm = flex.mat3_double()
    mersenne_twister = flex.mersenne_twister(seed=0)
    scitbx.random.set_random_seed(1234)
    rand_norm = scitbx.random.normal_distribution(mean=0, sigma=mosaic_spread_deg * math.pi/180.)
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(mosaic_domains)
    for m in mosaic_rotation:
        site = col(mersenne_twister.random_double_point_on_sphere())
        UMAT_nm.append( site.axis_and_angle_as_r3_rotation_matrix(m,deg=False) )

    if params.attenuation:
        detector_thick_mm = 0.032 # = 0 for Rayonix
        detector_thicksteps = 2 # should default to 1 for Rayonix, but set to 5 for CSPAD
        detector_attenuation_length_mm = 0.017 # default is silicon
    else:
        NotImplementedError("params.attenuation=False is not implemented")

    # get same noise each time this test is run
    seed = 1
    oversample=1
    polarization=1
    # this will become F000, marking the beam center
    default_F=0

    Amatrix_rot = (rand_ori*sqr(sfall_channels[0].unit_cell().orthogonalization_matrix())).transpose()
    xtal_shape=shapetype.Gauss_argchk # both crystal & RLP are Gaussian
    
    # rough approximation to water: interpolation points for sin(theta/lambda) vs structure factor
    water_bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.18,7.32),(0.2,6.75),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
    assert [a[0] for a in water_bg] == sorted([a[0] for a in water_bg])
    # rough approximation to air
    air_bg = flex.vec2_double([(0,14.1),(0.045,13.5),(0.174,8.35),(0.35,4.78),(0.5,4.22)])
    assert [a[0] for a in air_bg] == sorted([a[0] for a in air_bg])


    basic_params = (pixel_size_mm, Ncells_abc, shot_to_shot_wavelength_A, adc_offset_adu, mosaic_spread_deg, mosaic_domains, \
    distance_mm, UMAT_nm, detector_thick_mm, detector_thicksteps, detector_attenuation_length_mm, seed, oversample, \
    polarization, default_F, Amatrix_rot, xtal_shape, water_bg, air_bg, flux, wavlen, crystal)

    return basic_params

def tst_one_CPU(params, basic_params, sfall_channels, add_spots, use_background, num_pixels=3840):
    detpixels_slowfast=(num_pixels,num_pixels)
    
    pixel_size_mm, Ncells_abc, shot_to_shot_wavelength_A, adc_offset_adu, mosaic_spread_deg, mosaic_domains, \
    distance_mm, UMAT_nm, detector_thick_mm, detector_thicksteps, detector_attenuation_length_mm, seed, oversample, \
    polarization, default_F, Amatrix_rot, xtal_shape, water_bg, air_bg, flux, wavlen, crystal \
    = basic_params

    SIM = nanoBragg(detpixels_slowfast=detpixels_slowfast,pixel_size_mm=pixel_size_mm,Ncells_abc=Ncells_abc,
                    wavelength_A=shot_to_shot_wavelength_A,verbose=0)
    SIM.adc_offset_adu = adc_offset_adu
    SIM.mosaic_spread_deg = mosaic_spread_deg
    SIM.mosaic_domains = mosaic_domains
    SIM.distance_mm = distance_mm

    SIM.set_mosaic_blocks(UMAT_nm)

    SIM.detector_thick_mm = detector_thick_mm
    SIM.detector_thicksteps = detector_thicksteps
    SIM.detector_attenuation_length_mm = detector_attenuation_length_mm

    # get same noise each time this test is run
    SIM.seed = seed
    SIM.oversample = oversample
    SIM.wavelength_A = shot_to_shot_wavelength_A
    SIM.polarization = polarization
    # this will become F000, marking the beam center
    SIM.default_F = default_F
    SIM.Fhkl=sfall_channels[0] # instead of sfall_main

    SIM.Amatrix_RUB = Amatrix_rot
    print("SIM.Amatrix.RUB", SIM.Amatrix_RUB)
    #workaround for failing init_cell, use custom written Amatrix setter
    print("unit_cell_Adeg=",SIM.unit_cell_Adeg)
    print("unit_cell_tuple=",SIM.unit_cell_tuple)
    Amat = sqr(SIM.Amatrix).transpose() # recovered Amatrix from SIM
    # fastest option, least realistic
    SIM.xtal_shape=xtal_shape
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

    fluence_vec = []
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
        if add_spots:
            SIM.add_nanoBragg_spots()
        fluence_vec.append(SIM.fluence)

    nanoBragg_params = (SIM.phisteps, SIM.pix0_vector_mm, SIM.fdet_vector, SIM.sdet_vector, SIM.odet_vector, 
                        SIM.beam_vector, SIM.polar_vector, SIM.close_distance_mm, fluence_vec, 
                        SIM.beam_center_mm, SIM.spot_scale, SIM.curved_detector, SIM.point_pixel,
                        SIM.integral_form, SIM.nopolar)
    
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
        fluence_background = SIM.fluence
        SIM.add_background()
        SIM.Fbg_vs_stol = air_bg
        SIM.amorphous_sample_thick_mm = 10 # between beamstop and collimator
        SIM.amorphous_density_gcm3 = 1.2e-3
        SIM.amorphous_sample_molecular_weight_Da = 28 # nitrogen = N2
        SIM.add_background()
    else:
        fluence_background = None

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

        noise_params = (SIM.quantum_gain, SIM.detector_calibration_noise_pct, SIM.flicker_noise_pct, SIM.readout_noise_adu, \
        SIM.detector_psf_type, SIM.detector_psf_fwhm_mm, SIM.detector_psf_kernel_radius_pixels)
    else:
        noise_params = None
                
    return SIM.raw_pixels, nanoBragg_params, noise_params, fluence_background

def tst_one_pytorch(params, basic_params, Fhkl_mat_vec, add_spots, nanoBragg_params, noise_params, fluence_background, use_background, hkl_ranges, num_pixels=3840):
    h_max, h_min, k_max, k_min, l_max, l_min = hkl_ranges
    
    detpixels_slowfast=(num_pixels,num_pixels)
    pixel_size_mm, Ncells_abc, shot_to_shot_wavelength_A, adc_offset_adu, mosaic_spread_deg, mosaic_domains, \
    distance_mm, UMAT_nm, detector_thick_mm, detector_thicksteps, detector_attenuation_length_mm, seed, oversample, \
    polarization, default_F, Amatrix_rot, xtal_shape, water_bg, air_bg, flux, wavlen, crystal \
    = basic_params

    if xtal_shape==shapetype.Gauss_argchk:
        xtal_shape = 'GAUSS_ARGCHK'
    else:
        NotImplementedError("xtal_shape=%s is not implemented"%xtal_shape)

    phisteps, pix0_vector_mm, fdet_vector, sdet_vector, odet_vector, beam_vector, polar_vector, \
    close_distance_mm, fluence_vec, beam_center_mm, spot_scale, curved_detector, point_pixel, \
    integral_form, nopolar = nanoBragg_params

    spixels = num_pixels
    fpixels = num_pixels

    pixel_size = pixel_size_mm/1000
    roi_xmin = 0 
    roi_xmax = spixels
    roi_ymin = 0
    roi_ymax = fpixels
    maskimage = None
    detector_thickstep_mm = detector_thick_mm/detector_thicksteps
    detector_thickstep = detector_thickstep_mm/1000
    Odet = 3.200000e-05

    spindle_vector = torch.tensor([0,0,1]) # not used
    pix0_vector = torch.tensor(pix0_vector_mm)/1000
    fdet_vector = torch.tensor(fdet_vector)
    sdet_vector = torch.tensor(sdet_vector)
    odet_vector = torch.tensor(odet_vector)
    beam_vector = torch.tensor(beam_vector)
    polar_vector = torch.tensor(polar_vector)

    distance = distance_mm/1000
    close_distance = close_distance_mm/1000
    detector_thick = detector_thick_mm/1000
    detector_attnlen = detector_attenuation_length_mm/1000
    sources = 1
    source_X = torch.tensor([-10.000000])
    source_Y = torch.tensor([0.000000])
    source_Z  = torch.tensor([0.000000])
    source_I = torch.tensor([1.000000])
    dmin = 0.000000
    phi0 = 0.000000
    phistep = 0.000000
    phisteps = 1

    mosaic_umats = torch.tensor(UMAT_nm)

    ap = torch.tensor(Amatrix_rot[0:3])*1e-10
    bp = torch.tensor(Amatrix_rot[3:6])*1e-10
    cp = torch.tensor(Amatrix_rot[6:9])*1e-10

    a0 = ap # not used
    b0 = bp # not used
    c0 = cp # not used

    Na = Ncells_abc[0]
    Nb = Ncells_abc[1]
    Nc = Ncells_abc[2]
    fudge = 1
    V_cell = 8.093193e+06

    Xbeam = beam_center_mm[0]/1000.0
    Ybeam = beam_center_mm[1]/1000.0
    
    interpolate = False

    # loop over energies
    # raw_pixels = torch.zeros((num_pixels, num_pixels), requires_grad=True)
    raw_pixels_vec = []
    for x in range(len(flux)):
        print("Wavelength",x)
        # from channel_pixels function
        source_lambda = torch.tensor([wavlen[x]])*1e-10
        fluence = fluence_vec[x]
        Fhkl_input = Fhkl_mat_vec[x]

        if add_spots:
            raw_pixels_x = add_torchBragg_spots(spixels, 
                                fpixels,
                                phisteps,
                                mosaic_domains,
                                oversample,
                                pixel_size,
                                roi_xmin, roi_xmax, roi_ymin, roi_ymax,
                                maskimage, 
                                detector_thicksteps,
                                spot_scale, fluence,
                                detector_thickstep,
                                Odet,
                                fdet_vector, sdet_vector, odet_vector, pix0_vector,
                                curved_detector, distance, beam_vector, close_distance,
                                point_pixel,
                                detector_thick, detector_attnlen,
                                sources,
                                source_X, source_Y, source_Z, source_lambda,
                                dmin,phi0, phistep,
                                a0, b0, c0, ap, bp, cp, spindle_vector,
                                mosaic_spread_deg*math.pi/180,
                                mosaic_umats,
                                xtal_shape,
                                Na, Nb, Nc,
                                fudge,
                                integral_form,
                                V_cell,
                                Xbeam, Ybeam,
                                interpolate,
                                h_max, h_min, k_max, k_min, l_max, l_min,
                                Fhkl_input, default_F,
                                nopolar,source_I,
                                polarization,
                                polar_vector,
                                verbose=True,
                                use_numpy=False)
            raw_pixels_vec.append(raw_pixels_x)
    raw_pixels = torch.sum(torch.stack(raw_pixels_vec,axis=0),axis=0)



    # simulated crystal is only 125 unit cells (25 nm wide)
    # amplify spot signal to simulate physical crystal of 4000x larger: 100 um (64e9 x the volume)
    raw_pixels *= crystal.domains_per_crystal # must calculate the correct scale!

    source_lambda = torch.tensor([shot_to_shot_wavelength_A])*1e-10 # return to canonical energy for subsequent background

    if use_background:
        # add background of water
        stol_of = [-1e+99, -1e+98] + [a[0]*1e10 for a in water_bg] + [1e+98, 1e+99]
        Fbg_of = [water_bg[0][1],water_bg[0][1]] + [a[1] for a in water_bg] + [water_bg[-1][1], water_bg[-1][1]]

        stols = len(stol_of)
        stol_of = torch.tensor(stol_of)
        Fbg_of = torch.tensor(Fbg_of)
    
        Fmap_pixel = False
        override_source = -1
        amorphous_molecules= 30110708950000.000000

        detector_thickstep= 3.2e-05
        Odet= 0.000000e+00
        background_pixels_water, invalid_pixel = add_background(oversample, 
                                                            override_source,
                                                            sources,
                                                            spixels,
                                                            fpixels,
                                                            pixel_size,
                                                            roi_xmin, roi_xmax, roi_ymin, roi_ymax,
                                                            detector_thicksteps,
                                                            fluence_background, amorphous_molecules, 
                                                            Fmap_pixel, # bool override: just plot interpolated structure factor at every pixel, useful for making absorption masks
                                                            detector_thickstep, Odet, 
                                                            fdet_vector, sdet_vector, odet_vector, 
                                                            pix0_vector, curved_detector, distance, beam_vector,
                                                            close_distance, point_pixel, detector_thick, detector_attnlen,
                                                            source_I, source_X, source_Y, source_Z, source_lambda,
                                                            stol_of, stols, Fbg_of, nopolar, polarization, polar_vector,
                                                            verbose=True, use_numpy=False,
                                                            )

        # add background of air

        stol_of = [-1e+99, -1e+98] + [a[0]*1e10 for a in air_bg] + [1e+98, 1e+99]
        Fbg_of = [air_bg[0][1],air_bg[0][1]] + [a[1] for a in air_bg] + [air_bg[-1][1], air_bg[-1][1]]

        stols = len(stol_of)
        stol_of = torch.tensor(stol_of)
        Fbg_of = torch.tensor(Fbg_of)
    
        Fmap_pixel = False
        override_source = -1
        amorphous_molecules= 3613285074000.000488


        Odet= 3.200000e-05
        background_pixels_air, invalid_pixel = add_background(oversample, 
                                                            override_source,
                                                            sources,
                                                            spixels,
                                                            fpixels,
                                                            pixel_size,
                                                            roi_xmin, roi_xmax, roi_ymin, roi_ymax,
                                                            detector_thicksteps,
                                                            fluence_background, amorphous_molecules, 
                                                            Fmap_pixel, # bool override: just plot interpolated structure factor at every pixel, useful for making absorption masks
                                                            detector_thickstep, Odet, 
                                                            fdet_vector, sdet_vector, odet_vector, 
                                                            pix0_vector, curved_detector, distance, beam_vector,
                                                            close_distance, point_pixel, detector_thick, detector_attnlen,
                                                            source_I, source_X, source_Y, source_Z, source_lambda,
                                                            stol_of, stols, Fbg_of, nopolar, polarization, polar_vector,
                                                            verbose=True, use_numpy=False,
                                                            )

        raw_pixels = raw_pixels + background_pixels_water + background_pixels_air

    if params.noise or params.psf:
        if seed is not None:
            torch.manual_seed(seed)

        quantum_gain, detector_calibration_noise_pct, flicker_noise_pct, readout_noise_adu, \
        detector_psf_type, detector_psf_fwhm_mm, detector_psf_kernel_radius_pixels = noise_params

        raw_pixels = add_noise(raw_pixels, flicker_noise_pct/100., detector_calibration_noise_pct/100., 
                                torch.randn_like(raw_pixels), readout_noise_adu,
                                quantum_gain, adc_offset_adu, detector_psf_type.name.lower(), detector_psf_fwhm_mm, pixel_size_mm, 
                                detector_psf_kernel_radius_pixels, 'real_space')

    return raw_pixels

if __name__ == "__main__":
    params,options = parse_input()
    use_background = True
    add_spots = True
    num_pixels = 128 # change ranges for hkl if this is changed

    if num_pixels == 128:
        direct_algo_res_limit = 10.0 # need to make low enough to include all hkl on detector
        h_max= 11
        h_min= -11
        k_max= 22
        k_min= -22
        l_max= 30
        l_min= -30
    elif num_pixels == 512:
        direct_algo_res_limit = 8.0 # need to make low enough to include all hkl on detector
        h_max= 14
        h_min= -14
        k_max= 27
        k_min= -27
        l_max= 38
        l_min= -38
    elif num_pixels == 3840:
        direct_algo_res_limit = 1.85 # need to make low enough to include all hkl on detector
        h_max= 63
        h_min= -63
        k_max= 120
        k_min= -120
        l_max= 167
        l_min= -167
    else:
        NotImplementedError("num_pixels=%d is not implemented"%num_pixels)

    hkl_ranges = (h_max, h_min, k_max, k_min, l_max, l_min)
    sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit)
    basic_params = set_basic_params(params, sfall_channels)
    num_wavelengths = params.spectrum.nchannels
    Fhkl_mat_vec = get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_ranges, complex_output=False)

    raw_pixels, nanoBragg_params, noise_params, fluence_background = tst_one_CPU(params, basic_params, sfall_channels, add_spots, use_background, num_pixels=num_pixels)    
    raw_pixels_pytorch = tst_one_pytorch(params, basic_params, Fhkl_mat_vec, add_spots, nanoBragg_params, noise_params, fluence_background, use_background, hkl_ranges, num_pixels=num_pixels)

    if use_background:
        plt.figure(); plt.imshow(raw_pixels.as_numpy_array(), vmax=5.0e2, cmap='Greys');plt.colorbar();plt.savefig("raw_pixels.png")
        plt.figure(); plt.imshow(raw_pixels_pytorch.numpy(), vmax=5.0e2, cmap='Greys');plt.colorbar();plt.savefig("raw_pixels_torch.png")
    else:
        plt.figure(); plt.imshow(raw_pixels.as_numpy_array(), vmax=10e-5, cmap='Greys');plt.colorbar();plt.savefig("raw_pixels.png")
        plt.figure(); plt.imshow(raw_pixels_pytorch.numpy(), vmax=10e-5, cmap='Greys');plt.colorbar();plt.savefig("raw_pixels_torch.png")
    
    plt.figure(); plt.imshow(raw_pixels.as_numpy_array(), cmap='Greys');plt.colorbar();plt.savefig("raw_pixels_no_cap.png")
    plt.figure(); plt.imshow(raw_pixels_pytorch.numpy(), cmap='Greys');plt.colorbar();plt.savefig("raw_pixels_torch_no_cap.png")

    
