import numpy as np
def CSP(xs,zs,w0,x,k0,dz,Nz):
    ''' function that compute the initial reduced field u asuming a CSP as the pattern'''
    u = np.zeros(Nz,dtype='complex')
    # Position of the CSP in the complex plane
    x0 = 0.5*k0*w0**2
    xpos = x - xs + 1j*x0
    for iz in range(Nz):
        zpos = iz*dz - zs
        # Compute the range from the CSP to the point M
        r2 = xpos**2 + zpos**2 + 1j*(1e-15)
        rtilde = np.sqrt(np.abs(r2))*np.exp(1j*(np.angle(r2)+(1-np.sign(np.angle(r2)))*np.pi)/2.0)
        # Reduced field u
        u[iz] = 1j/(4*np.sqrt(np.pi*rtilde))*np.exp(-1j*k0*rtilde-k0*x0)
    norm = np.max(np.abs(u))
    # Normalized to avoid very low values
    u = u/norm
    return u, norm

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('MACOSX')
    import matplotlib.pyplot as plt

    # Parameters (source)
    xs = -100
    zs = 50
    w0 = 5
    # Parameters (Propagation)
    x = 0
    f = 300e6
    c = 3e8
    wavelength = c/f
    k0 = 2*np.pi*f/c
    zmax = 256
    # Discretization
    dz = 0.5*wavelength
    z = np.arange(0,zmax,dz)
    Nz = int(zmax/dz)
    # Initial field
    u,_ = CSP(xs, zs, w0, x, k0, dz, Nz)
    plt.figure()
    plt.plot(20*np.log10(np.abs(u)+1e-15),z)
    plt.show()