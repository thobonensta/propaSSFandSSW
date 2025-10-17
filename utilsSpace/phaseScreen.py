import numpy as np
def phasescreenwide(dx,n,k0):
    ''' Compute the phase screen for the wide angle PWE '''
    L = np.exp(-1j*k0*(n-1)*dx)
    return L

