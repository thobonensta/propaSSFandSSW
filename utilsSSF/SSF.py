import math
import numpy as np
from utilsSpace.ground import calculatedTwavenumber,FresnelCoeff, addImageField
from utilsSSF.transform import FFT, IFFT
from tqdm import tqdm
from utilsRelief.shiftRelief import shift_relief_propa, shift_relief_postpropa

def waDSSF(u0,x0,zs,k0,epsr1,epsr2,dx,Nx,Nz,Nim,Napo,P,L,A,zt,polar,condG):
    ''' function that performs the DSSF (wide-angle) method to solve the PWE
    '''

    # define an array of zero to store each iteration over x
    usave = np.zeros((Nx,Nz),dtype='complex')
    if condG == 'PEC' or condG =='Dielectric':
        usave[0,:] = u0[:-Napo]
    else:
        usave[0,:] = u0[Napo:-Napo]

    ux = u0

    # Shift from one step to another needed for the terrain
    shiftT = np.diff(zt)


    for ix in tqdm(range(1,Nx)):
        xpos = ix*dx + x0
        # Compute the Fresnel coefficient for the ground
        thetaI = math.atan(xpos/zs)
        kiz,ktz = calculatedTwavenumber(k0,epsr1,epsr2,thetaI)
        if condG == 'PEC':
            R = -1
            ux = addImageField(ux, Nim, R)
        elif condG == 'Dielectric':
            R = FresnelCoeff(epsr1,epsr2,kiz,ktz,polar)
            ux = addImageField(ux, Nim, R)

        # If descending relief -> zero bottom
        if shiftT[ix - 1] < 0: # Descending stair
            ux = shift_relief_propa(ux, shiftT[ix - 1])

        # Go in the spectral domain to propagate in a free-space layer
        Ux = FFT(ux)
        # One step of propagation in the spectral domain
        Uxdx = P*Ux
        # Go back in the spatial domain
        uxdxFS = IFFT(Uxdx)

        # pop the image field in the ground
        if condG == 'PEC' or condG=='Dielectric':
            uxdxFS = uxdxFS[Nim:]

        if shiftT[ix - 1]> 0: # Ascending stair
            uxdxFS = shift_relief_propa(uxdxFS, shiftT[ix - 1])

        # account for refraction and apodisation in the spatial domain
        ux = A*L*uxdxFS

        if condG == 'PEC' or condG == 'Dielectric':
            usave[ix,:] = ux[:-Napo]
        else:
            usave[ix, :] = ux[Napo:-Napo]

        # shift back to the physical domain
        tr = zt[ix]
        usave[ix, :] = shift_relief_postpropa(usave[ix, :], tr)

    return usave

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
    Nx = 5
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
    P = propa_FS_widediscrete(k0, dx, dz, Nim+Nz+Napo)
    L = phasescreenwide(dx,n,k0)
    A = HanningWindowUp(zmax,dz,Nz,Napo)

    # Compute the SSF propagation
    usave = waDSSF(u, x, zs, k0, 1, 1, dx, Nx, Nz, Nim, Napo, P, L, A, polar='TE', condG='PEC')

    plt.figure()
    plt.plot(20*np.log10(np.abs(uTrue*normp/norm)+1e-15),z,label='True')
    plt.plot(20*np.log10(np.abs(usave[-1,:])+1e-15),z,'--',color='orange',label='DSSF')
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
    usave = waDSSF(u0, x, zs, k0, 1, 1, dx, Nx, Nz, Nim, Napo, P, L, A, polar='TE', condG='None')

    plt.figure()
    plt.plot(20 * np.log10(np.abs(uTrue * normp / norm) + 1e-15), z, label='True')
    plt.plot(20 * np.log10(np.abs(usave[-1, :]) + 1e-15), z, '--', color='orange', label='DSSF')
    plt.plot(20 * np.log10(np.abs(usave[-1, :] - uTrue * normp / norm) + 1e-15), z, '--', color='orange', label='diff')
    plt.grid()
    plt.legend()
    vmax = np.max(20 * np.log10(np.abs(uTrue * normp / norm) + 1e-15)) + 1
    vmin = vmax - 70
    plt.xlim([vmin, vmax])
    print('diff max : ', np.max(20 * np.log10(np.abs(usave[-1, :] - uTrue * normp / norm) + 1e-15)))
    plt.show()

