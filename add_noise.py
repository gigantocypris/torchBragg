import torch
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
from apply_psf import apply_psf

def add_noise(raw_pixels,
              flicker_noise,
              calibration_noise,
              calibration_noise_sample, # sample from normal distribution the same shape as raw_pixels
              readout_noise, # adu units
              quantum_gain,
              adc_offset,
              psf_type,
              psf_fwhm, # width of the actual psf in the same units as pixel_size
              pixel_size,
              psf_radius, # pixel width of the computed kernel
              convolution_type, # 'real_space' or 'fourier_space'
              true_poisson=True,
              ):
    
    """ Add noise to the raw_pixels image
    1. Add flicker noise (multiplicative Gaussian)
    2. Add Poisson noise (using Gaussian approximation to the Poisson distribution)
    3. Add calibration noise (additive Gaussian that is the same for all shots in the experiment)
    4. Apply PSF
    5. Add read out noise:
        Convert photon signal to read out units (photon signal is floatimage[i]): adu = floatimage[i]*quantum_gain + adc_offset
        Additive Gaussian readout noise, readout_noise is in adu units: adu += readout_noise * image_deviates.gaussdev( &seed );
    """

    loc = raw_pixels
    scale = 0 # standard deviation

    # add flicker noise
    # approximation because adding a single sample of flicker noise to the mean of the distribution
    if flicker_noise > 0:
        loc *= (1.0 + flicker_noise * torch.randn_like(loc))

    # add Poisson noise (Gaussian approximation to the Poisson distribution)
    # approximation to the original nanoBragg model because a Gaussian approximation

    if true_poisson:
        loc = Poisson(loc).sample()
    else:
        scale += torch.sqrt(loc)

    # add calibration noise
    # calibration is same from shot to shot, but varies from pixel to pixel
    if calibration_noise > 0:
        loc *= (1.0 + calibration_noise * calibration_noise_sample)

    # implement PSF by convolving loc with a kernel, scale is unchanged

    if psf_type != 'unknown':
        loc = apply_psf(loc,
                        psf_type,
                        psf_fwhm/pixel_size, # width of the actual psf
                        psf_radius,
                        convolution_type = convolution_type,
                       )

    # convert loc and scale to adu units
    scale *= quantum_gain
    loc *= quantum_gain
    loc += adc_offset

    # add readout noise
    scale += readout_noise

    return(Normal(loc, scale).sample())

