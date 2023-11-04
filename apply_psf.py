import numpy as np
import torch
import torch.nn.functional as F
import math

def get_kernel(psf_type, fwhm_pixels, psf_radius):
    
    # convert fwhm to "g" distance : fwhm = sqrt((2**(2./3)-1))/2*g # XXX this equation is incorrect
    g = fwhm_pixels * 0.652383013252053

    # calculate the PSF
    spixel_range = np.arange(-psf_radius, psf_radius+1) # slow pixels
    fpixel_range = np.arange(-psf_radius, psf_radius+1) # fast pixels

    kernel = torch.zeros((len(spixel_range), len(fpixel_range)))

    for spixel in spixel_range:
        for fpixel in fpixel_range:
            
            rsq = spixel**2 + fpixel**2
            if(rsq > psf_radius*psf_radius):
                pass
            else:
                if psf_type == 'gauss':
                    kernel[spixel, fpixel] = integrate_gauss_over_pixel(spixel, fpixel, fwhm_pixels, 1.0)
                elif psf_type == 'fiber':
                    kernel[spixel, fpixel] = integrate_fiber_over_pixel(spixel, fpixel, g, 1.0)
                else:
                    raise ValueError('Unknown PSF type')
    return(kernel)

def apply_psf(raw_pixels,
              psf_type,
              fwhm_pixels, # width of the actual psf
              psf_radius, # pixel width of the computed kernel
              ):

    pixel_sum_start = torch.sum(raw_pixels)
    kernel = get_kernel(psf_type, fwhm_pixels, psf_radius)
    
    # convolve with PSF
    raw_pixels = raw_pixels[None,None,:,:] # add batch and channel dims
    kernel = kernel[None,None,:,:] # add batch and channel dims
    raw_pixels = F.conv2d(raw_pixels, kernel, padding='same') # XXX do convolution in Fourier space

    # lost pixels
    pixel_sum_end = torch.sum(raw_pixels)

    # add back lost pixels
    lost_pixels = (pixel_sum_start - pixel_sum_end) / raw_pixels.numel()
    raw_pixels += lost_pixels
    return(raw_pixels)

# integral of Gaussian fwhm=1 integral=1
def ngauss2D_integ(x, y):
    return 0.125*(math.erf(2.0*x*np.sqrt(np.log(2.0)))*math.erf(y*np.sqrt(np.log(16.0)))*np.sqrt(np.log(16.0)/np.log(2.0)))

# unit volume integrated over a pixel, fwhm = 1
def ngauss2D_pixel(x, y, pix):
    return ngauss2D_integ(x+pix/2.,y+pix/2.)-ngauss2D_integ(x+pix/2.,y-pix/2.)-ngauss2D_integ(x-pix/2.,y+pix/2.)+ngauss2D_integ(x-pix/2.,y-pix/2.)

def integrate_gauss_over_pixel(x, y, fwhm, pix):
    return ngauss2D_pixel(x/fwhm,y/fwhm,pix/fwhm)

def fiber2D_integ(x, y, g):
    return np.arctan((x*y)/(g*np.sqrt(g*g + x*x + y*y)))/2.0/np.pi

def fiber2D_pixel(x, y, g, pix):
     return fiber2D_integ(x+pix/2.,y+pix/2.,g)-fiber2D_integ(x+pix/2.,y-pix/2.,g)-fiber2D_integ(x-pix/2.,y+pix/2.,g)+fiber2D_integ(x-pix/2.,y-pix/2.,g)

def integrate_fiber_over_pixel(x, y, g, pix):
    return fiber2D_pixel(x,y,g,pix)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    kernel = get_kernel('gauss', 5, 10)
    plt.figure();plt.imshow(kernel);plt.savefig("kernel_gauss.png")

    kernel = get_kernel('fiber', 5, 10)
    plt.figure();plt.imshow(kernel);plt.savefig("kernel_fiber.png")
    breakpoint()