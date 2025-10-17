import numpy as np
def propa_FS_widediscrete(k0,dx,dz,Nz):
    ''' compute the free-space propagator for the discrete wide-angle PWE '''
    kx = np.zeros(Nz,dtype='complex')
    # Init the vector q_z in [0,N-1]
    vect_qz = np.linspace(0,Nz-1,num=Nz,endpoint=True,dtype=int)
    # discrete spectral step kz (calculated from Fourier spectral diagonalization of the operator)
    kz = 2/dz*np.sin(np.pi*vect_qz/Nz)
    # calculate kx and take into account the sign of sqrt (evanescent or not)
    kx2 = k0**2 - kz**2
    kx[kx2>=0] = np.sqrt(kx2[kx2>=0])
    kx[kx2<0] = -1j*np.sqrt(kx2[kx2<0])
    # DSSF propagator
    P = np.exp(-1j*dx*(kx-k0))
    return P

if __name__ == '__main__':
    from utilsSource.ComplexSourcePoint import CSP
    from utilsSSF.transform import FFT, IFFT
    import matplotlib
    matplotlib.use('MACOSX')
    import matplotlib.pyplot as plt

    # Parameters (source)
    xs = -100
    zs = 128
    w0 = 5
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
    Nz = int(zmax/dz)
    # Initial field
    u,norm = CSP(xs, zs, w0, x, k0, dz, Nz)

    # Propagation over dx
    uTrue,normp = CSP(xs, zs, w0, x+dx, k0, dz, Nz)

    P = propa_FS_widediscrete(k0, dx, dz, Nz)
    U = FFT(u)
    Up = P*U
    up = IFFT(Up)

    plt.figure()
    plt.plot(20*np.log10(np.abs(uTrue*normp/norm)+1e-15),z,label='True')
    plt.plot(20*np.log10(np.abs(up)+1e-15),z,'--',color='orange',label='step SSF')
    plt.plot(20*np.log10(np.abs(up-uTrue*normp/norm)+1e-15),z,'--',color='orange',label='diff')
    plt.grid()
    plt.legend()
    vmax = np.max(20*np.log10(np.abs(uTrue*normp/norm)+1e-15))+1
    vmin = vmax-70
    plt.xlim([vmin,vmax])
    plt.show()




