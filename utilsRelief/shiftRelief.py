import numpy as np
def shift_relief_propa(arr, num):
    ''' function that performs the shift of relief to propagate
    Inputs:
    arr: numpy array. The array (field) to be shifted (accumulated shift from the last relief)
    num: int. shift in Nz that corresponds to the difference of relief along the way
    Output:
    result: arr that has been shifted for the relief
    '''
    result = np.zeros_like(arr)
    if num < 0:  # donward
        result[-num:] = arr[:num]

    elif num > 0:  # upward
        result[:-num] = arr[num:]

    else:
        result[:] = arr
    return result

def shift_relief_postpropa(u_field, ii_relief):
    ''' function that shifts back the field in the physical domain'''
    if ii_relief == 0:
        u_field_shifted = u_field
    else:
        u_field_shifted = np.zeros_like(u_field)
        u_field_shifted[ii_relief:] = u_field[:-ii_relief]
    return u_field_shifted