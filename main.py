from utilsSource.ComplexSourcePoint import CSP
from utilsSpace.phaseScreen import phasescreenwide
from utilsSpace.apodisation import HanningWindowUp
from utilsSSF.SSF import waDSSF
from utilsSSF.propaFreeSpace import propa_FS_widediscrete
from utilslSSW.lSSW import lSSW
from utilsRefraction.refractivity import linearRefractivity,trilinearRefractivity
import numpy as np
from utilsRelief.staircazeModel import model_relief
import time
import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    c = 3e8
    # Simulation parameters
    f = 300e6# Simulation frequency [Hz]
    wavelength = c/f
    k0 = 2*np.pi/wavelength
    xmax = 15000 # Maximal range for the simulation [m]
    zmax = 512 # Maximal altitude for the simulation [m]
    # Source parameters
    xs = - 50 # position of the source in x [m]
    zs = 70 # position of the source in z from the ground [m]
    w0 = 5*wavelength # width of the CSP
    # Discretization
    dx = 50*wavelength
    dz = 0.5*wavelength
    Nx = int(xmax/dx)
    Nz = int(zmax/dz)
    NimSSF = Nz
    c = 0.1
    NimSSW = int(c*Nz)
    Napo = Nz
    # Source field (CSP placed at [xs,zs] with a width w0)
    x0 = 0
    u0,_ = CSP(xs,zs,w0,x0,k0,dz,Nz+Napo)
    # Polarisation
    polar = 'TE'
    # Ground parameters
    mu0 = 4 * np.pi * 1e-7
    epsilon0 = 1 / (c ** 2 * mu0)
    condG = 'Dielectric' # condition of the ground [PEC or Dielectric or None]
    epsr1 = 1.0
    eps2 = 20
    sig2 = 0.02
    epsr2 = eps2 - 1j * sig2 / (2 * np.pi * f * epsilon0)
    # Refractive index
    refrac = 'trilinear'
    M0 = 330
    c0 = 0.118
    c1= -0.5
    zb = 20
    zt = 50
    if refrac == 'freespace':
        n = np.ones(Nz)
    elif refrac == 'linear':
        n = linearRefractivity(Nz,M0,c0,dz)
    elif refrac == 'trilinear':
        n = trilinearRefractivity(dz,Nz,zb,zt,zmax,M0,c0,c1,c0)


    ntot = np.zeros(Nz+Napo)
    ntot[:Nz] = n
    ntot[Nz:] = n[-1]

    # Terrain
    xterrain = [0, 5000, 8000, 12000, xmax]
    zterrain = [0, 0, 100, 0, 0]
    xt, zt = model_relief(0,Nx, dx, dz, xterrain, zterrain)
    plt.figure()
    plt.plot(xt, zt*dz)
    plt.show()
    ############################################################################################

    # Defining the space operator
    L = phasescreenwide(dx,ntot,k0)
    # Defining the apodisation layer
    A = HanningWindowUp(zmax,dz,Nz,Napo)

    ############################################################################################

    # Propagation with SSF
    t0 = time.perf_counter()
    P = propa_FS_widediscrete(k0,dx,dz,NimSSF+Nz+Napo)
    ussf = waDSSF(u0,x0,zs,k0,epsr1,epsr2,dx,Nx,Nz,NimSSF,Napo,P,L,A,zt,polar,condG)
    print('Time SSF : ',time.perf_counter()-t0)

    # Propagation with SSW
    family = 'sym6'
    level = 2
    Vs = 1e-3*np.max(np.abs(u0))
    Vp = 1e-4
    remaining = NimSSW % (2 ** level)
    if remaining:  # if not zero
        NimSSW += (2 ** level) - remaining
    t0 = time.perf_counter()
    ussw = lSSW(u0,x0,zs,k0,epsr1,epsr2,dx,Nx,dz,Nz,NimSSW,Napo,L,A,zt,polar,condG,family,level,Vs,Vp)
    print('Time SSW : ',time.perf_counter()-t0)

    plt.figure()
    ussfdB = 20*np.log10(np.abs(ussf.T)+1e-15)
    vmax = np.max(ussfdB)
    vmin = vmax - 50
    extent = [x0,xmax,0,zmax]
    plt.imshow(ussfdB,extent=extent,aspect='auto',origin='lower',vmax=vmax,vmin=vmin,cmap='gnuplot',interpolation='none')
    plt.colorbar()
    plt.xlabel('Range [m]')
    plt.ylabel('Altitude [m]')
    plt.figure()
    usswdB = 20*np.log10(np.abs(ussw.T)+1e-15)
    vmax = np.max(usswdB)
    vmin = vmax - 50
    extent = [x0,xmax,0,zmax]
    plt.imshow(usswdB,extent=extent,aspect='auto',origin='lower',vmax=vmax,vmin=vmin,cmap='gnuplot',interpolation='none')
    plt.colorbar()
    plt.xlabel('Range [m]')
    plt.ylabel('Altitude [m]')
    plt.figure()
    err = 20*np.log10(np.abs(ussf.T-ussw.T)+1e-15)
    plt.imshow(err,extent=extent,aspect='auto',origin='lower',cmap='gnuplot',interpolation='none')
    print('Err max : ', np.max(err))
    plt.colorbar()
    plt.xlabel('Range [m]')
    plt.ylabel('Altitude [m]')
    plt.show()




