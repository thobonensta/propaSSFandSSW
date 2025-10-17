import numpy as np
def model_relief(x0,Nx,dx,dz,xterrain,zterrain):
    ''' function that compute the list of altitude of terrain using a staircaze model
    The idea is to model the terrain as a sequence of rectangles
    Inputs :
    x0 : float (usually 0). Xpos for the first vertical of initial condition
    Nx: int. Number of discretization points for the propagation axis
    dx: float. step over the propagation axis
    dz: float. step over the altitude
    xterrain: list of points where a changed of altitude of terrain is observed in m
    zterrain: list of the corresponding change of altitude in m
    Outputs:
    x_list: list of discretized points over the x axis
    z_list: list of interpolated altitude of the terrain
    '''
    x_list = np.zeros(Nx)
    for ii_x in range(0,Nx):
        x_list[ii_x] = x0 + ii_x*dx
    # vector of the altitudes of the relief
    z_relief = np.interp(x_list,xterrain,zterrain)
    # transformation into N points
    z_list = np.round(z_relief / dz).astype('int')
    return x_list, z_list

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('MACOSX')
    import matplotlib.pyplot as plt

    # Model a triangle obstacle between 5000 and 7000m with a peak altitude of 500m
    xterrain = [0, 5000, 6000, 7000, 10000]
    zterrain = [0, 0, 500, 0, 0]
    Nx = 100
    dx = 100
    dz = 0.5
    xt, zt = model_relief(0,Nx, dx, dz, xterrain, zterrain)
    plt.figure()
    plt.plot(xt, zt*dz)
    plt.show()