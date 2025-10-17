import math
import numpy as np
from utilsSpace.ground import calculatedTwavenumber,FresnelCoeff, addImageField
from utilsSSF.transform import FFT, IFFT


def waDSSF(u0,x0,zs,k0,epsr1,epsr2,dx,Nx,Nz,Nim,Napo,P,L,A,polar,condG):

    usave = np.zeros((Nx,Nz),dtype='complex')
    if condG == 'PEC' or condG =='Dielectric':
        usave[0,:] = u0[:-Napo]
    else:
        usave[0,:] = u0[Napo:-Napo]

    ux = u0

    for ix in range(1,Nx):
        xpos = ix*dx + x0
        thetaI = math.atan(xpos/zs)
        kiz,ktz = calculatedTwavenumber(k0,epsr1,epsr2,thetaI)
        if condG == 'PEC':
            R = -1
            ux = addImageField(ux, Nim, R)
        elif condG == 'Dielectric':
            R = FresnelCoeff(epsr1,epsr2,kiz,ktz,polar)
            ux = addImageField(ux, Nim, R)


        Ux = FFT(ux)
        Uxdx = P*Ux
        uxdxFS = IFFT(Uxdx)

        if condG == 'PEC' or condG=='Dielectric':
            uxdxFS = uxdxFS[Nim:]

        ux = A*L*uxdxFS

        if condG == 'PEC' or condG == 'Dielectric':
            usave[ix,:] = ux[:-Napo]
        else:
            usave[ix, :] = ux[Napo:-Napo]


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

