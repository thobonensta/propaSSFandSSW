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