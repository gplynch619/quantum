'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This program simulates the quantum dynamics of certain quantum harmonic oscillator (QHO) 
states via an implementation of the spectral method (see Feit et. al. 1982). It uses the
built-in FFT that comes with numpy for its Fourier Transforms. This is currently all done
in one dimension but the generalization to 3 dimensions should be straightforward, with
special attention paid to parallelization. Units are SI.

'''
import os
import math
import numpy as np
import constants

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def main():
    iterations=[]
    for filename in os.listdir('8000T'): 
        data=np.load('2000T/'+filename)
        energy_spec=data['E']
        param=data['meta'] #meta[0]=dt, meta[1]=ntime
        ts=data['ts']        
        
        peaks_indx,_=find_peaks(np.abs(energy_spec))
        
        peaks_e=2*np.pi*ts[peaks_indx]
        iterations.append([param[0], param[1], peaks_e[-1]])
    iterations.sort(key=lambda x: x[1])
    iterations=np.array(iterations)
    
    dt=iterations[:,0]
    ntime=iterations[:,1]
    E=iterations[:,2]

    plt.rc('text', usetex=True)
    fig, ax = plt.subplots()
    ax.plot(dt, E, linestyle='None', marker='.')
    ax.set_xlabel("$dt$")
    ax.set_ylabel("E")
    ax.set_title("T=8000, estimated E vs. time step")
    ax.axhline(y=-0.5, linestyle='--', color='C7')
    plt.show()
    #fig.savefig("adw_spectrum.png")
if __name__=="__main__":
    main()

