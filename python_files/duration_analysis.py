'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This program simulates the quantum dynamics of certain quantum harmonic oscillator (QHO) 
states via an implementation of the spectral method (see Feit et. al. 1982). It uses the
built-in FFT that comes with numpy for its Fourier Transforms. This is currently all done
in one dimension but the generalization to 3 dimensions should be straightforward, with
special attention paid to parallelization. Units are SI.

'''
import math
import numpy as np
import constants

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from quantum_system import *
from harmonic import *

def main():
    
    m=constants.M
    omega=constants.OMEGA
    hbar=constants.HBAR      #Reduced Planck                      

    N=2**9     #number of spatial samples
    x,p = initialize_grid(-10, N, 10.0, endpoint_mode=True)
    psi=energy_eigenstate(0)           #from harmonic.py
    v=harmonic_potential()         #from harmonic.py
    v_n=v(x)
    qs = System(x, psi(x),v_n)
    
    durations=np.arange(start=4000, stop=13000, step=50)
    dts=[]
    peaklocs=[]
    T=4000
    for ntime in durations.astype(int): 
        dt= float(T)/ntime
        ts=np.fft.fftfreq(ntime, dt)
        ts=np.fft.fftshift(ts)
        cor_func_t=[]
        for step in ts:
            qs.time_evolve(dt, 1)
            psi=qs._get_psi_x()
            psi0=qs._get_initial_psi()
            temp_val=np.trapz(np.conjugate(psi0)*psi, x=qs.x)
            cor_func_t.append(temp_val)
        cor_func_t=np.array(cor_func_t)

        energy_spec=np.fft.fft(cor_func_t)/ntime
        energy_spec=np.fft.fftshift(energy_spec)    

        peaks_indx,_=find_peaks(np.abs(energy_spec))
        peaks_e=2*np.pi*ts[peaks_indx]
        peaklocs.append(peaks_e[-1])
        dts.append(dt)
        print("Finished time step {}".format(dt))
        
    dts=np.array(dts)
    peaklocs=np.array(peaklocs)
    f='analysis_data/{}zoomin3.csv'.format(4000) 
    with open(f, 'w') as filename:
        filename.write('dt,E,T\n')
        for i, nstep in enumerate(durations):
            filename.write('{0},{1},{2}\n'.format(dts[i], peaklocs[i], nstep))

    plt.rc('font', family='serif')
    fig, ax = plt.subplots()
    
    ax.plot(dts, peaklocs, 'o')
    ax.set_title(r"$N_t$={}".format(ntime))
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"Energy estimate")
    plt.show()

    #fig.savefig("adw_spectrum.png")
if __name__=="__main__":
    main()

