import numpy as np
def HanningWindowUp(zmax,dz,Nz,Napo):
    ''' function that compute the apodisation operator assuming a Hanning window (Up)'''
    A = np.ones(Nz+Napo)
    zpos = np.arange(zmax,2*zmax,dz)-zmax
    window = (1+np.cos(np.pi*zpos/zmax))/2
    A[-Napo:]=window
    return A

def HanningWindowUpDown(zmax,dz,Nz,Napo):
    ''' function that compute the apodisation operator assuming a Hanning window (Up and Down)'''
    A = np.ones(Napo+Nz+Napo)
    zpos = np.arange(zmax,int(Napo/Nz+1)*zmax,dz)-zmax
    window = (1+np.cos(np.pi*zpos/(int(Napo/Nz)*zmax)))/2
    A[-Napo:] = window
    A[:Napo] = window[::-1]
    return A

def PMLEdgesUpDown(dz,Nz, Npad, slope=0.8):
    ''' function that performs apodisation using a PML strategy UP
    based on a first order approximation at the top boundary of Sommerfeld condition
    '''
    alpha = np.zeros(Npad+Nz+Npad)
    alpha[:Npad] = slope*((Npad - np.arange(Npad))/Npad)**2
    alpha[-Npad:] = slope*((np.arange(Npad)+1)/Npad)**2
    A = np.exp(-alpha * dz)
    return A

def PMLEdgesUp(dz,Nz, Npad, slope=0.8):
    ''' function that performs apodisation using a PML strategy Up and Down
    based on a first order approximation at the top and bottom boundary of Sommerfeld condition
    '''

    alpha = np.zeros(Nz+Npad)
    alpha[-Npad:] = slope*((np.arange(Npad)+1)/Npad)**2
    A = np.exp(-alpha * dz)
    return A