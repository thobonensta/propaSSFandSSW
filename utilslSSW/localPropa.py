import matplotlib.pyplot as plt
import pywt
import numpy as np
from utilsSSF.propaFreeSpace import propa_FS_widediscrete
from utilsSSF.transform import FFT, IFFT
from utilslSSW.transformFWT import compressedFWT
def localPropagators(dx,dz,family,level,k0,Vp):
    lst_wavelets = waveletEachLevel(dx,dz,family,level)

    locP = initLocPropa(level)

    nbT = nbrTranslations(level)

    for il in range(level+1):
        Tl = nbT[il]
        uw = lst_wavelets[il]
        Nsupp = len(uw)
        P = propa_FS_widediscrete(k0,dx,dz,Nsupp)
        Uw = FFT(uw)
        Uwdx = P*Uw
        uwdx = IFFT(Uwdx)
        for it in range(Tl):
            t = it*(2**(level+1-il))
            uwdxt = np.roll(uwdx,t)
            ptmp = compressedFWT(uwdxt,family,level,Vp*np.max(np.abs(uwdxt)))
            locP[il][it] = ptmp
    return locP




def waveletEachLevel(dx,dz,family,level):

    lst_wavelets = [[] for _ in range(level+1)]

    # Generating the scaling function to propagate
    for il in range(level+1):
        n = waveletsize(il, level)
        np = increaseWSizePropaWA(n, dx, dz, level)
        lst_wavelets[il] = createWavelet(np,family,il,level)

    return lst_wavelets

def createWavelet(ns,family,il,level):
    u = np.zeros(ns+ns,dtype='complex')
    U = pywt.wavedec(u,family,'per',level)
    if il == 0:
        i0 = int(ns/2**(level))
    else:
        i0 = int(ns/2**(level-il+1))
    U[il][i0] = 1
    uw = pywt.waverec(U,family,'per')
    return uw

def waveletsize(il,level):
    n = 7
    l = il
    if il == 0:
        l = 1
    n *= (2**(level-l))
    return n

def increaseWSizePropaWA(n,dx,dz,level):
    nadd = int(np.ceil(dx*np.sin(np.pi/2)/dz))
    n = n + nadd
    rem = n%(2**level)
    if rem:
        n += 2**level - rem
    return n

def nbrTranslations(level):
    nbT = np.zeros(level+1,dtype=int)
    for il in range(level+1):
        level = il
        if level == 0:
            level = 1
        Tl = 2**(level-1)
        nbT[il] = Tl
    return nbT

def initLocPropa(level):
    nbT = nbrTranslations(level)
    LocP = [[] for _ in range(level+1)]
    for il in range(level+1):
        Tl = nbT[il]
        LocP[il] = [[] for _ in range(Tl)]
    return LocP


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('MACOSX')
    import matplotlib.pyplot as pl

    level = 2
    dx = 100
    dz = 0.5
    lst_wavelets = waveletEachLevel(dx,dz,'sym6',level)

    for il in range(level+1):
        plt.figure()
        plt.plot(np.real(lst_wavelets[il]))
    plt.show()

    f = 300e6
    c = 3e8
    k0 = 2*np.pi*f/c
    locP = library_generation(dx,dz,'sym6',2,k0,1e-5)
    locPt = localPropagators(dx,dz,'sym6',2,k0,1e-5)


    for ip in range(len(locP)):
        ptmp = locP[ip]
        ptmp2 = locPt[ip]
        for it in range(len(ptmp)):
            uwp = pywt.waverec(ptmp[it],'sym6','per')
            uwp2 = pywt.waverec(ptmp2[it],'sym6','per')

            plt.figure()
            plt.plot(20*np.log10(np.abs(uwp-uwp2)+1e-15))
    plt.show()



