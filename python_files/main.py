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
import yaml
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from quantum_system import *
from importlib import import_module

def load_config(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
def main():
    config=load_config(argv[1])
   
    ##Set up time steps##
    T=config['T']
    ntime=config['ntime']
    dt=float(T)/ntime

    ##Set up grid##
    N=config['N']
    xstart=config['xstart']
    xstop_or_dx=config['xstop_or_dx']
    grid_flag=config['endpoint_mode']

    x,p = initialize_grid(-20, N, 20.0, endpoint_mode=grid_flag)

    ##Set up wavefunction##
    psi_library=import_module('wavefunctions')
    psi=getattr(psi_library, d['wavefunction']['type'])(d['wavefunction']['params'])
    psi_n=psi(x)

    v=getattr(v_library, d['potential']['type'])(d['type']['params'])
    v_n=v(x)

    v_n=v_n+shift
    # shift so zero is at top of well?


    #Begin evolution of system#
    qs = System(x, psi_n,v_n)
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

    fig,ax=plt.subplots()
    ax.plot(x, qs._get_psi_x().real, x, qs._get_psi(x).imag)
    ax.plot(x, v_n)

if __name__=="__main__":
    main()

