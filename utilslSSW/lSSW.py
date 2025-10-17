import pywt
import numpy as np
from utilslSSW.localPropa import localPropagators, nbrTranslations
import math
from utilsSpace.ground import calculatedTwavenumber,FresnelCoeff, addImageField
from utilslSSW.transformFWT import compressedFWT
from scipy.signal import convolve
from tqdm import tqdm
from utilsRelief.shiftRelief import shift_relief_propa, shift_relief_postpropa


def lSSW(u0,x0,zs,k0,epsr1,epsr2,dx,Nx,dz,Nz,Nim,Napo,L,A,zt,polar,condG,family,level,Vs,Vp):
    ''' function that performs the local SSW algorithm
    We solve the PWE iteratively (as an ODE) using SSW (local version)
    '''

    # define an array of zero to store each iteration over x
    usave = np.zeros((Nx,Nz),dtype='complex')
    if condG == 'PEC' or condG =='Dielectric':
        usave[0,:] = u0[:-Napo]
    else:
        usave[0,:] = u0[Napo:-Napo]

    # Compute the local operators associated to the wavelet
    locP =  localPropagators(dx,dz,family,level,k0,Vp)
    # Translations needed for each level depending on dilations
    nbrT = nbrTranslations(level)

    ux = u0

    # Shift from one step to another needed for the terrain
    shiftT = np.diff(zt)


    for ix in tqdm(range(1,Nx)):
        xpos = ix * dx + x0
        # Compute the Fresnel coefficient for the ground
        thetaI = math.atan(xpos / zs)
        kiz, ktz = calculatedTwavenumber(k0, epsr1, epsr2, thetaI)
        if condG == 'PEC':
            R = -1
            ux = addImageField(ux, Nim, R)
        elif condG == 'Dielectric':
            R = FresnelCoeff(epsr1, epsr2, kiz, ktz, polar)
            ux = addImageField(ux, Nim, R)

        # If descending relief -> zero bottom
        if shiftT[ix - 1] < 0: # Descending stair
            ux = shift_relief_propa(ux, shiftT[ix - 1])

        # Pass in the wavelet domain
        Ux = compressedFWT(ux,family,level,Vs)
        # Propagate in the wavelet domain
        UxdxFS = propaWOneStep(Ux,locP,family,level,len(ux),nbrT)
        # Come back in the space domain
        uxdxFS = pywt.waverec(UxdxFS,family,'per')

        # Pop the image field
        if condG == 'PEC' or condG == 'Dielectric':
            uxdxFS = uxdxFS[Nim:]

        # If ascending relief -> zero up
        if shiftT[ix - 1]> 0: # Ascending stair
            uxdxFS = shift_relief_propa(uxdxFS, shiftT[ix - 1])

        # Refraction and apodisation in the spatial domain
        ux = A * L * uxdxFS

        if condG == 'PEC' or condG == 'Dielectric':
            usave[ix, :] = ux[:-Napo]
        else:
            usave[ix, :] = ux[Napo:-Napo]

        # shift back to the physical domain
        tr = zt[ix]
        usave[ix, :] = shift_relief_postpropa(usave[ix, :], tr)

    return usave


def propaWOneStep(Ux,locP,family,level,Nz,nbrT):
    ''' function that computes one step of propagation in the wavelet domain
    This is done by convoluting the local operators over the non-zero wavelet
    coefficients. One need to choose the right operators depending on the level
    and the dilations.
    '''
    # --- Initialization ---
    tmpZ = np.zeros(Nz, dtype=complex)
    Uxdx = pywt.wavedec(tmpZ, family, 'per', level)

    # --- Main computation ---
    for il in range(level + 1):
        nT = nbrT[il]
        Uxl = Ux[il]

        # Precompute shifted filters for all local patterns at this level
        locPlt_shifted = [
            [np.roll(arr, -1) for arr in sublist]
            for sublist in locP[il]
        ]

        for it in range(nT):
            Uxlm = Uxl[it::nT]
            locPlt_it = locPlt_shifted[it]

            for il2 in range(level + 1):
                nT2 = nbrT[il2]
                nzdilate = nT2 * Uxlm.size

                # --- Dilation ---
                Uxdxldilate = np.zeros(nzdilate, dtype=complex)
                Uxdxldilate[::nT2] = Uxlm

                # --- Convolution ---
                Uxdx[il2] += convolve(Uxdxldilate, locPlt_it[il2], mode='same', method='auto')

    return Uxdx



if __name__ == '__main__':
    from utilsSource.ComplexSourcePoint import CSP
    from utilsSSF.propaFreeSpace import propa_FS_widediscrete
    from utilsSpace.phaseScreen import phasescreenwide
    from utilsSpace.apodisation import HanningWindowUp, HanningWindowUpDown

    import matplotlib
    matplotlib.use('MACOSX')
    import matplotlib.pyplot as plt

    # Parameters (source)
    xs = -100
    zs = 128
    w0 = 7
    # Parameters (Propagation)
    x = 0
    f = 300e6
    c = 3e8
    wavelength = c/f
    k0 = 2*np.pi*f/c
    zmax = 256
    # Discretization
    dx = 100*wavelength
    dz = 0.5*wavelength
    z = np.arange(0,zmax,dz)
    Nx = 2
    Nz = int(zmax/dz)
    Nim = Nz
    Napo = Nz
    # Parameters (Refraction)
    n = np.ones(Nz+Napo)
    # Initial field
    u,norm = CSP(xs, zs, w0, x, k0, dz, Nz+Napo)

    # Propagation over dx
    uTrue,normp = CSP(xs, zs, w0, x+(Nx-1)*dx, k0, dz, Nz)

    # Compute all the operators beforehand
    L = phasescreenwide(dx,n,k0)
    A = HanningWindowUp(zmax,dz,Nz,Napo)

    # Compute the SSF propagation
    usave = lSSW(u, x, zs, k0, 1, 1, dx, Nx, dz,Nz, Nim, Napo, L, A, 'TE', 'PEC','sym6', 1, 1e-10, 1e-10)

    plt.figure()
    plt.plot(20*np.log10(np.abs(uTrue*normp/norm)+1e-15),z,label='True')
    plt.plot(20*np.log10(np.abs(usave[-1,:])+1e-15),z,'--',color='orange',label='SSW')
    plt.plot(20*np.log10(np.abs(usave[-1,:]-uTrue*normp/norm)+1e-15),z,'--',color='orange',label='diff')
    plt.grid()
    plt.legend()
    vmax = np.max(20*np.log10(np.abs(uTrue*normp/norm)+1e-15))+1
    vmin = vmax-70
    plt.xlim([vmin,vmax])
    print('diff max : ',np.max(20*np.log10(np.abs(usave[-1,:]-uTrue*normp/norm)+1e-15)))
    plt.show()

    # Second test without ground  = apo up and down for propagation in space
    # Parameters (Refraction)
    n = np.ones(Napo+Nz + Napo)
    # Initial field
    u, norm = CSP(xs, zs, w0, x, k0, dz, Nz + Napo)
    u0 = np.zeros(Napo+Nz+Napo,dtype='complex')
    u0[Napo:] = u
    # Propagation over dx
    uTrue, normp = CSP(xs, zs, w0, x + (Nx - 1) * dx, k0, dz, Nz)

    # Compute all the operators beforehand
    P = propa_FS_widediscrete(k0, dx, dz, Napo + Nz + Napo)
    L = phasescreenwide(dx, n, k0)
    A = HanningWindowUpDown(zmax, dz, Nz, Napo)

    # Compute the SSF propagation
    usave = lSSW(u0, x, zs, k0, 1, 1, dx, Nx, dz,Nz, Nim, Napo, L, A, 'TE', 'None','sym6', 2, 1e-5, 1e-5)

    plt.figure()
    plt.plot(20 * np.log10(np.abs(uTrue * normp / norm) + 1e-15), z, label='True')
    plt.plot(20 * np.log10(np.abs(usave[-1, :]) + 1e-15), z, '--', color='orange', label='SSW')
    plt.plot(20 * np.log10(np.abs(usave[-1, :] - uTrue * normp / norm) + 1e-15), z, '--', color='orange', label='diff')
    plt.grid()
    plt.legend()
    vmax = np.max(20 * np.log10(np.abs(uTrue * normp / norm) + 1e-15)) + 1
    vmin = vmax - 70
    plt.xlim([vmin, vmax])
    print('diff max : ', np.max(20 * np.log10(np.abs(usave[-1, :] - uTrue * normp / norm) + 1e-15)))
    plt.show()

