from utils import which_package

def sincg_vectorized(x, N, prefix):
    sincg_i = prefix.sin(x*N)/prefix.sin(x)
    sincg_i[x==0] = N
    return sincg_i

def sinc3_vectorized(x, prefix):
    """Fourier transform of a sphere"""
    sinc3_i = 3.0*(prefix.sin(x)/x-prefix.cos(x))/(x*x)
    sinc3_i[x==0] = 1.0
    return sinc3_i
