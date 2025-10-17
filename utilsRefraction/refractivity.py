import numpy as np
def linearRefractivity(Nz,M0,c0,dz):
    ''' model for a linear refractivity'''
    M_index = np.zeros(Nz)
    for jj_z in range(0,Nz):
    # Refractivity is expressed in M-units
        M_index[jj_z] = M0 + c0 *jj_z *dz
    return 1.0 + M_index * 1e-6

def trilinearRefractivity(dz,Nz,zb,zt,zmax,M0,c0,c1,c2):
    ''' model for a tri-linear refractivity'''
    config_alt = [0, zb, zb + zt, zmax]
    Mb = M0 + zb * c0
    Mt = Mb + zt * c1
    M_max = Mt + (zmax - (zb - zt)) * c2
    config_Mindex = [M0, Mb, Mt, M_max]
    # Refractivity is expressed in M-units
    M_index = np.interp(np.arange(0, Nz * dz, dz), config_alt, config_Mindex)
    return 1.0 + M_index * 1e-6

