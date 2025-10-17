import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import numpy as np

def FFT(u_in):
    ''' function that computes the FFT of the field u_in
    normalized by its length
    INPUT : u_in: numpy array (size N)
    OUTPUT : fft of u_in: array of same size
    '''
    N = len(u_in)
    fft_u_in = 1/np.sqrt(N)*fft(u_in)
    return fft_u_in

def IFFT(fft_u_in):
    ''' function that computes the IFFT of the fft of u_in
    normalized by its length
    INPUT : fft_u_in: numpy array (size N)
    OUTPUT : u_in: array of same size
    '''
    N = len(fft_u_in)
    u_in = np.sqrt(N)*ifft(fft_u_in)
    return u_in

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('MACOSX')
    import matplotlib.pyplot as plt

    x = np.linspace(0,2*np.pi,128)
    u_in = np.sin(x)
    fft_u = FFT(u_in)
    u = IFFT(fft_u)

    fig, axs = plt.subplots(2)
    axs[0].plot(x,np.real(u_in),'-',color='black',label='u')
    axs[0].plot(x,np.real(u),'--', color='orange',label='u after FFT and IFFT')
    axs[0].legend()
    axs[1].plot(np.abs(fft_u))
    plt.show()