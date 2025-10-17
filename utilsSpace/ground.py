import numpy as np

def calculatedTwavenumber(k0,epsr1,epsr2,thetaI):
    ''' function that computes the reflected and transmitted wave number at the ground'''
    k0 = k0*np.sqrt(epsr1)
    ksol = k0*np.sqrt(epsr2)
    kiz = -k0*np.cos(thetaI)
    ktx = k0*np.sin(thetaI)
    ktz = np.sqrt(ksol**2-ktx**2)
    if np.imag(ktz)<0:
        ktz = -ktz
    return kiz,ktz

def FresnelCoeff(epsr1,epsr2,kiz,ktz,polar):
    ''' function that compute the Fresnel coefficient at the ground'''
    if polar == 'TM':
        R = (epsr2*kiz-epsr1*ktz)/(epsr2*kiz+epsr1*ktz)
    elif polar == 'TE':
        R = (kiz-ktz)/(kiz+ktz)
    return R

def addImageField(u,Nim,R):
    ''' function that add the image field -- image theorem'''
    Nz = len(u)
    uim = np.zeros(Nim+Nz,dtype='complex')
    uim[Nim:] = u
    uim[0:Nim] = R * np.flip(u[1:Nim+1])
    return uim

if __name__ == '__main__':
    from utilsSource.ComplexSourcePoint import CSP
    import matplotlib
    matplotlib.use('MACOSX')
    import matplotlib.pyplot as plt

    # Parameters (source)
    xs = -100
    zs = 20
    w0 = 5
    # Parameters (Propagation)
    x = 0
    f = 300e6
    c = 3e8
    wavelength = c / f
    k0 = 2 * np.pi * f / c
    zmax = 128
    # Discretization
    dx = 100 * wavelength
    dz = 0.5 * wavelength
    z = np.arange(0, zmax, dz)
    zim = np.arange(-zmax, zmax, dz)
    Nz = int(zmax / dz)
    # Initial field
    u, norm = CSP(xs, zs, w0, x, k0, dz, Nz)
    u[0] = 0
    R = -1
    uim = addImageField(u,Nz,R)

    plt.figure()
    plt.plot(20 * np.log10(np.abs(u) + 1e-15), z, label='u')
    plt.plot(20 * np.log10(np.abs(uim) + 1e-15), zim, '--', color='orange', label='uim')
    plt.grid()
    plt.legend()
    vmax = np.max(20 * np.log10(np.abs(u) + 1e-15)) + 1
    vmin = vmax - 70
    plt.xlim([vmin, vmax])
    plt.show()



