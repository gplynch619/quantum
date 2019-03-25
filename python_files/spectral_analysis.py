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
from scipy.signal import argrelextrema

from quantum_system import *
#from harmonic import *
from asym_double_well import *

def main():
    ##Defining Parameters##
    
    m=constants.M
    omega=constants.OMEGA
    hbar=constants.HBAR      #Reduced Planck                      

    #Initialize timing info (how long to simulate for, and smallest time step)#
    dt=.005  #time step
    ntime=16384      #number of time steps
    sampling=1.0/dt   #how many samples per unit (f in hertz)
    T=ntime*dt        #total time
    ####

    #Initialize grids#
    #This is from System.py#
    N=2**9     #number of spatial samples
    x,p = initialize_grid(-20, N, 20.0, endpoint_mode=True)
    #####


    #Initialize wavefunction and potential#
#   psi=coherent_state(1)              #from harmonic.py
#    psi=energy_eigenstate(0)           #from harmonic.py
    psi=gaussian_super()            #from asym

#    v=harmonic_potential()         #from harmonic.py
    v=asym_double_well_potential()#from asym
    v_n=v(x)

    zeros=np.where(np.diff(np.signbit(v_n)))[0]
    v_n[:zeros[0]+1]=0
    v_n[zeros[1]+1:]=0
    #these lines turn the unbounded well into a finite well
    shift=0 #132.7074997 
    v_n=v_n+shift
    # shift so zero is at top of well?


    #Begin evolution of system#
    qs = System(x, psi(x)+shift,v_n)
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

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(2,1)
    #dx/sqrt(2pi) * psi(x_n)* exp(-ik0x_n) <--> \tilde(psi)(k_m)exp(-im*x0*dk)
    line1, = ax[0].plot(qs.x, qs.v_x, label=r'v(x)')
    line2, = ax[0].plot(qs.x, 25*qs._get_initial_psi()-80, label='$\psi (x)$ scaled')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('v(x)')
    ax[0].set_xlim([-10, 10])
   
    ax[0].legend()

    ax[1].plot(-2*np.pi*ts, np.abs(energy_spec))
#    ax[1].scatter(peaks, np.abs(energy_spec)[peaks], marker='x')
    
    ax[1].set_yscale('log') 
    ax[1].set_xlabel(r"Energy ($\hbar \omega$)")
    ax[1].set_ylabel(r"Spectral power")
    ax[1].set_xlim([-150, 20])
    plt.show()

    fig.savefig("adw_spectrum.png")
if __name__=="__main__":
    main()

