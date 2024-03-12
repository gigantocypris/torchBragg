import os
import math
import torch
from LS49.spectra.generate_spectra import spectra_simulation
from LS49.sim.step4_pad import microcrystal
from LS49 import legacy_random_orientations
from exafel_project.kpp_utils.ferredoxin import basic_detector_rayonix
from simtbx.nanoBragg import shapetype
from scitbx.array_family import flex
import scitbx
from scitbx.matrix import sqr,col
from diffraction_vectorized import add_torchBragg_spots
from add_background_vectorized import add_background
from add_noise import add_noise
torch.set_default_dtype(torch.float64)

def set_all_params(params, sfall_channels, device):
    spectra = spectra_simulation()
    crystal = microcrystal(Deff_A = params.crystal.Deff_A, length_um = params.crystal.length_um, beam_diameter_um = 1.0) # assume smaller than 10 um crystals
    # random_orientation = legacy_random_orientations(100)[0]
    random_orientation = legacy_random_orientations(1)[0]
    
    DETECTOR = basic_detector_rayonix()
    PANEL = DETECTOR[0]
    
    iterator = spectra.generate_recast_renormalized_image_parameterized(image=0,params=params)
    rand_ori = sqr(random_orientation)

    wavlen, flux, shot_to_shot_wavelength_A = next(iterator) # list of lambdas, list of fluxes, average wavelength
    wavlen = torch.tensor(wavlen.as_numpy_array()[:,None], device=device)
    assert shot_to_shot_wavelength_A > 0 # wavelength varies shot-to-shot

    # use crystal structure to initialize Fhkl array
    N = crystal.number_of_cells(sfall_channels[0].unit_cell())
    
    pixel_size_mm = PANEL.get_pixel_size()[0]
    pixel_size = pixel_size_mm/1000
    # Ncells_abc=(N,N,N)
    Na = N
    Nb = N
    Nc = N

    adc_offset_adu = 0 # Do not offset by 40
    mosaic_spread_deg = 0.05 # interpreted by UMAT_nm as a half-width stddev
                             # mosaic_domains setter MUST come after mosaic_spread_deg setter
    mosaic_domains = int(os.environ.get("MOS_DOM"))
    distance_mm = PANEL.get_distance()
    distance = distance_mm/1000

    UMAT_nm = flex.mat3_double()
    mersenne_twister = flex.mersenne_twister(seed=0)
    scitbx.random.set_random_seed(1234)
    rand_norm = scitbx.random.normal_distribution(mean=0, sigma=mosaic_spread_deg * math.pi/180.)
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(mosaic_domains)
    for m in mosaic_rotation:
        site = col(mersenne_twister.random_double_point_on_sphere())
        UMAT_nm.append(site.axis_and_angle_as_r3_rotation_matrix(m,deg=False))

    mosaic_umats = torch.tensor(UMAT_nm, device=device)
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
        detector_thick = detector_thick_mm/1000
        detector_thicksteps = 2 # should default to 1 for Rayonix, but set to 5 for CSPAD
        detector_attenuation_length_mm = 0.017 # default is silicon
        detector_attnlen = detector_attenuation_length_mm/1000
        detector_thickstep_mm = detector_thick_mm/detector_thicksteps
        detector_thickstep = detector_thickstep_mm/1000
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


    phisteps = 1
    pix0_vector_mm = torch.tensor((141.7, 5.72*10, -5.72*10), device=device) # detector origin, change to get different ROI #XXX
    # pix0_vector_mm = torch.tensor((141.7, 169.04799999999997, -169.04799999999997), device=device) # detector origin, change to get different ROI # original ROI for 3840 pixels
    # pix0_vector_mm = torch.tensor((141.7, 5.72, -5.72), device=device) # detector origin, change to get different ROI # original ROI for 128 pixels
    pix0_vector = pix0_vector_mm/1000
    fdet_vector = torch.tensor((0.0, 0.0, 1.0), device=device)
    sdet_vector = torch.tensor((0.0, -1.0, 0.0), device=device)
    odet_vector = torch.tensor((1.0, 0.0, 0.0), device=device)
    beam_vector = torch.tensor((1.0, 0.0, 0.0), device=device)
    polar_vector = torch.tensor((0.0, 0.0, 1.0), device=device)
    close_distance_mm = 141.7
    close_distance = close_distance_mm/1000
    fluence_vec = [1.374132034195255e+19, 1.0686071593132872e+19]
    beam_center_mm = (5.675999999999999, 5.675999999999999)
    Xbeam = beam_center_mm[0]/1000.0
    Ybeam = beam_center_mm[1]/1000.0
    spot_scale = 1.0
    curved_detector = False
    point_pixel = False
    integral_form = False
    nopolar = False
    Odet = 3.200000e-05
    sources = 1
    source_X = torch.tensor([-10.000000], device=device)
    source_Y = torch.tensor([0.000000], device=device)
    source_Z  = torch.tensor([0.000000], device=device)
    source_I = torch.tensor([1.000000], device=device)
    dmin = 0.000000
    phi0 = 0.000000
    phistep = 0.000000
    phisteps = 1
    ap = torch.tensor(Amatrix_rot[0:3], device=device)*1e-10
    bp = torch.tensor(Amatrix_rot[3:6], device=device)*1e-10
    cp = torch.tensor(Amatrix_rot[6:9], device=device)*1e-10
    fudge = 1
    V_cell = 8.093193e+06
    interpolate = False
    fluence_background = 1.1111111111111111e+23

    source_lambda_background = torch.tensor([shot_to_shot_wavelength_A], device=device)*1e-10 # return to canonical energy for subsequent background

    # water background params
    stol_of_water = [-1e+99, -1e+98] + [a[0]*1e10 for a in water_bg] + [1e+98, 1e+99]
    Fbg_of_water = [water_bg[0][1],water_bg[0][1]] + [a[1] for a in water_bg] + [water_bg[-1][1], water_bg[-1][1]]

    stols_water = len(stol_of_water)
    stol_of_water = torch.tensor(stol_of_water, device=device)
    Fbg_of_water = torch.tensor(Fbg_of_water, device=device)

    Fmap_pixel = False
    override_source = -1
    amorphous_molecules_water= 30110708950000.000000

    detector_thickstep_water= 3.2e-05
    Odet_water= 0.000000e+00

    # air background params
    stol_of_air = [-1e+99, -1e+98] + [a[0]*1e10 for a in air_bg] + [1e+98, 1e+99]
    Fbg_of_air = [air_bg[0][1],air_bg[0][1]] + [a[1] for a in air_bg] + [air_bg[-1][1], air_bg[-1][1]]

    stols_air = len(stol_of_air)
    stol_of_air = torch.tensor(stol_of_air, device=device)
    Fbg_of_air = torch.tensor(Fbg_of_air, device=device)

    amorphous_molecules_air = 3613285074000.000488
    Odet_air = 3.200000e-05
    
    # noise params
    quantum_gain = 1.0
    detector_calibration_noise_pct = 1.0
    flicker_noise_pct = 0.0
    readout_noise_adu = 1.0
    detector_psf_type = shapetype.Unknown
    detector_psf_fwhm_mm = 0.0
    detector_psf_kernel_radius_pixels = 0

    all_params = (pixel_size, Na, Nb, Nc, shot_to_shot_wavelength_A, adc_offset_adu, mosaic_spread_deg, mosaic_domains, \
                  distance, mosaic_umats, detector_thick, detector_thickstep, detector_thicksteps, detector_attnlen, seed, oversample, \
                  polarization, default_F, ap, bp, cp, xtal_shape, flux, wavlen, crystal, \
                  phisteps, pix0_vector, fdet_vector, sdet_vector, odet_vector, beam_vector, polar_vector, close_distance, fluence_vec, \
                  Xbeam, Ybeam, spot_scale, curved_detector, point_pixel, integral_form, nopolar, Odet, sources, source_X, source_Y, source_Z, source_I, \
                  dmin, phi0, phistep, phisteps, fudge, V_cell, interpolate, fluence_background, source_lambda_background, stol_of_water, Fbg_of_water, stols_water, \
                  Fmap_pixel, override_source, amorphous_molecules_water, detector_thickstep_water, Odet_water, stol_of_air, Fbg_of_air, \
                  stols_air, amorphous_molecules_air, Odet_air, quantum_gain, detector_calibration_noise_pct, \
                  flicker_noise_pct, readout_noise_adu, detector_psf_type, detector_psf_fwhm_mm, detector_psf_kernel_radius_pixels)

    return all_params

def forward_sim(params, all_params, Fhkl_mat_vec, add_spots, 
                use_background, hkl_ranges, device, num_pixels):
    h_max, h_min, k_max, k_min, l_max, l_min = hkl_ranges

    pixel_size, Na, Nb, Nc, shot_to_shot_wavelength_A, adc_offset_adu, mosaic_spread_deg, mosaic_domains, \
    distance, mosaic_umats, detector_thick, detector_thickstep, detector_thicksteps, detector_attnlen, seed, oversample, \
    polarization, default_F, ap, bp, cp, xtal_shape, flux, wavlen, crystal, \
    phisteps, pix0_vector, fdet_vector, sdet_vector, odet_vector, beam_vector, polar_vector, close_distance, fluence_vec, \
    Xbeam, Ybeam, spot_scale, curved_detector, point_pixel, integral_form, nopolar, Odet, sources, source_X, source_Y, source_Z, source_I, \
    dmin, phi0, phistep, phisteps, fudge, V_cell, interpolate, fluence_background, source_lambda_background, stol_of_water, Fbg_of_water, stols_water, \
    Fmap_pixel, override_source, amorphous_molecules_water, detector_thickstep_water, Odet_water, stol_of_air, Fbg_of_air, \
    stols_air, amorphous_molecules_air, Odet_air, quantum_gain, detector_calibration_noise_pct, \
    flicker_noise_pct, readout_noise_adu, detector_psf_type, detector_psf_fwhm_mm, detector_psf_kernel_radius_pixels \
    = all_params

    if xtal_shape==shapetype.Gauss_argchk:
        xtal_shape = 'GAUSS_ARGCHK'
    else:
        NotImplementedError("xtal_shape=%s is not implemented"%xtal_shape)

    # loop over energies
    # raw_pixels = torch.zeros((num_pixels, num_pixels), requires_grad=True)
    raw_pixels_vec = []
    for x in range(len(flux)):
        print("Wavelength",x)
        # from channel_pixels function
        source_lambda = wavlen[x]*1e-10
        fluence = fluence_vec[x]
        Fhkl_input = Fhkl_mat_vec[x]

        if add_spots:
            raw_pixels_x = add_torchBragg_spots(num_pixels, num_pixels, phisteps, mosaic_domains,
                                                oversample, pixel_size, 0, num_pixels, 0, num_pixels,
                                                None, detector_thicksteps, spot_scale, fluence, detector_thickstep,
                                                Odet, fdet_vector, sdet_vector, odet_vector, pix0_vector,
                                                curved_detector, distance, beam_vector, close_distance,
                                                point_pixel, detector_thick, detector_attnlen, sources,
                                                source_X, source_Y, source_Z, source_lambda, dmin,phi0, phistep,
                                                None, None, None, ap, bp, cp, None, mosaic_spread_deg*math.pi/180,
                                                mosaic_umats, xtal_shape, Na, Nb, Nc, fudge, integral_form, V_cell,
                                                Xbeam, Ybeam, interpolate, h_max, h_min, k_max, k_min, l_max, l_min,
                                                Fhkl_input, default_F, nopolar, source_I, polarization, polar_vector,
                                                device, verbose=True, use_numpy=False)
            raw_pixels_vec.append(raw_pixels_x)
    raw_pixels = torch.sum(torch.stack(raw_pixels_vec,axis=0),axis=0)



    # simulated crystal is only 125 unit cells (25 nm wide)
    # amplify spot signal to simulate physical crystal of 4000x larger: 100 um (64e9 x the volume)
    raw_pixels *= crystal.domains_per_crystal # must calculate the correct scale!

    if use_background:
        # add background of water
        background_pixels_water, invalid_pixel = add_background(oversample, override_source, sources,
                                                                num_pixels, num_pixels, pixel_size,
                                                                0, num_pixels, 0, num_pixels, detector_thicksteps,
                                                                fluence_background, amorphous_molecules_water, 
                                                                Fmap_pixel, # bool override: just plot interpolated structure factor at every pixel, useful for making absorption masks
                                                                detector_thickstep_water, Odet_water, 
                                                                fdet_vector, sdet_vector, odet_vector, 
                                                                pix0_vector, curved_detector, distance, beam_vector,
                                                                close_distance, point_pixel, detector_thick, detector_attnlen,
                                                                source_I, source_X, source_Y, source_Z, source_lambda_background,
                                                                stol_of_water, stols_water, Fbg_of_water, nopolar, polarization, polar_vector,
                                                                verbose=True, use_numpy=False, device=device,
                                                            )

        # add background of air
        background_pixels_air, invalid_pixel = add_background(oversample, override_source, sources,
                                                              num_pixels, num_pixels, pixel_size, 0, num_pixels, 0, num_pixels,
                                                              detector_thicksteps, fluence_background, amorphous_molecules_air, 
                                                              Fmap_pixel, # bool override: just plot interpolated structure factor at every pixel, useful for making absorption masks
                                                              detector_thickstep, Odet_air, fdet_vector, sdet_vector, odet_vector, 
                                                             pix0_vector, curved_detector, distance, beam_vector,
                                                             close_distance, point_pixel, detector_thick, detector_attnlen,
                                                             source_I, source_X, source_Y, source_Z, source_lambda_background,
                                                             stol_of_air, stols_air, Fbg_of_air, nopolar, polarization, polar_vector,
                                                             verbose=True, use_numpy=False, device=device,
                                                             )

        raw_pixels = raw_pixels + background_pixels_water + background_pixels_air

    if params.noise or params.psf:
        if seed is not None:
            torch.manual_seed(seed)

        raw_pixels = add_noise(raw_pixels, flicker_noise_pct/100., detector_calibration_noise_pct/100., 
                                torch.randn_like(raw_pixels), readout_noise_adu,
                                quantum_gain, adc_offset_adu, detector_psf_type.name.lower(), detector_psf_fwhm_mm, pixel_size*1000, 
                                detector_psf_kernel_radius_pixels, 'real_space')

    return raw_pixels