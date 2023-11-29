from utils import which_package
def find_pixel_pos_vectorized(Fdet_mat, Sdet_mat, Odet_vec, fdet_vector, sdet_vector, odet_vector, pix0_vector,
                   curved_detector, distance, beam_vector, use_numpy):
    """ construct detector subpixel position in 3D space """
    prefix, new_array = which_package(use_numpy)

    # pixel_pos_mat is [Fdet_mat.shape[0], Fdet_mat.shape[1], len(Odet_vec), 3]
    pixel_pos_mat = Fdet_mat[:,:,None,None]*fdet_vector[None,None,None,:]+Sdet_mat[:,:,None,None]*sdet_vector[None,None,None,:]+Odet_vec[None,None,:,None]*odet_vector[None,None,None,:]+pix0_vector[None,None,None,:] 


    if curved_detector:
        raise NotImplementedError
    return pixel_pos_mat 
