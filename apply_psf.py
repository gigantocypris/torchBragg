import numpy as np
import torch
import torch.nn.functional as F
import math

def get_kernel(psf_type, fwhm_pixels, psf_radius):
    
    # convert fwhm to "g" distance : fwhm = sqrt((2**(2./3)-1))/2*g # XXX this equation is incorrect
    g = fwhm_pixels * 0.652383013252053

    # calculate the PSF
    spixel_range = np.arange(-psf_radius, psf_radius) # slow pixels
    fpixel_range = np.arange(-psf_radius, psf_radius) # fast pixels

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
              convolution_type = 'real_space' # 'real_space' or 'fourier_space'
              ):

    pixel_sum_start = torch.sum(raw_pixels)
    if convolution_type == 'fourier_space':
        if (raw_pixels.shape[0] != raw_pixels.shape[1])  or raw_pixels.shape[0] % 2:
            raise ValueError('Number of pixels must be even for convolution in Fourier space and need square shape')
        psf_radius = raw_pixels.shape[0]//2 # make the kernel the same size as the image
    kernel = get_kernel(psf_type, fwhm_pixels, psf_radius)
    
    if convolution_type == 'fourier_space':
        # do the convolution in Fourier space
        kernel_FFT = torch.fft.fft2(kernel)
        raw_pixels_FFT = torch.fft.fft2(torch.fft.fftshift(raw_pixels))
        raw_pixels_FFT = raw_pixels_FFT * kernel_FFT
        raw_pixels = torch.fft.ifftshift(torch.fft.ifft2(raw_pixels_FFT))
        raw_pixels = torch.abs(raw_pixels) # remove the imaginary part which should be close to zero
    else:
        # convolve with PSF in real space
        raw_pixels = raw_pixels[None,None,:,:] # add batch and channel dims
        kernel = torch.fft.fftshift(kernel)
        kernel = kernel[None,None,:,:] # add batch and channel dims
        raw_pixels = F.conv2d(raw_pixels, kernel, padding='same') # XXX do convolution in Fourier space
        raw_pixels = raw_pixels[0,0,:,:] # remove batch and channel dims
        
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
    kernel = get_kernel('gauss', 16, 64)
    plt.figure();plt.imshow(torch.fft.fftshift(kernel));plt.savefig("kernel_gauss.png")

    kernel = get_kernel('fiber', 16, 64)
    plt.figure();plt.imshow(torch.fft.fftshift(kernel));plt.savefig("kernel_fiber.png")

    raw_pixels = torch.zeros((128,128))
    raw_pixels[64,64] = 1.0
    psf_type = 'gauss'
    fwhm_pixels = 16
    psf_radius = 64

    raw_pixels = apply_psf(raw_pixels, psf_type, fwhm_pixels, psf_radius, convolution_type = 'real_space')
    plt.figure();plt.imshow(raw_pixels);plt.savefig("convolved_pixels_0.png")

    raw_pixels = torch.zeros((128,128))
    raw_pixels[64,64] = 1.0

    raw_pixels = apply_psf(raw_pixels, psf_type, fwhm_pixels, psf_radius, convolution_type = 'fourier_space')
    plt.figure();plt.imshow(raw_pixels);plt.savefig("convolved_pixels_1.png")

    breakpoint()