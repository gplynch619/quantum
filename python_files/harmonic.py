'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This program simulates the quantum dynamics of certain quantum harmonic oscillator (QHO) 
states via an implementation of the spectral method (see Feit et. al. 1982). It uses the
built-in FFT that comes with numpy for its Fourier Transforms. This is currently all done
in one dimension but the generalization to 3 dimensions should be straightforward, with
special attention paid to parallelization. Units are natural units. 

TODO:
1. Energy eigenstates and coherent states
2. Would it be better to make a wavefunction class that gets initialized to something?
3. How to initialize grid
'''
import math
import numpy as np
import matplotlib.pyplot as plt
from quantum_system import *

from scipy.special import hermite
from scipy import signal

def harmonic_potential(x):
    return 0.5*x*x #1/2m*w^2*x^2 but ignoring constants for n.0001avefunction states##
def energy_eigenstate(n): #Factory function to generate psi
    ''' This function is used to generate the energy eigenstates of the QHO.
    Parameters
    -------
    energy_level : an integer indicating which excited state should be generated.
    0 is the ground state and n is the nth excided state

    Returns
    -------
    psi(x) : A function that when evaluted gives the value of the indicated state at the position.
    '''
    def psi(x):
        return (1.0/np.sqrt((2**n)*math.factorial(n)))*np.pi**(-1.0/4)*np.exp((-x**2)/2)*hermite(n)(x)
    
    return psi

def coherent_state(a):
    
    def psi(x):
        return (1/np.pi**0.25)*np.exp(-(np.sqrt(0.5)*x-a)**2 + 0.5*(a**2- np.abs(a)**2))
    return psi

def main():
    ##Defining Parameters##
    
    dt=.01   #arbitrary 'small' dt 
    hbar=1      #Reduced Planck                      
    N=2**11     #number of spatial samples
    t_max = 25
    Nt_steps=int(t_max/dt)
    ####
    
    #Initialize position and momentum grid#
    x0=-6.0   
    xf=6.0     #start and end x's. This ranges from [-6,6) in discrete steps
    dx = (xf-x0)/N    
    x=[x0+n*dx for n in range(N)] #1D grid of N steps with length dx
    x=np.array(x)

    p0= -0.5*N*dx #Satisfies Nyquist limit, same as -pi/dx
    dp=2*np.pi/(N*dx)
    p=[p0+m*dp for m in range(N)]
    p=np.array(p)

    #Initialize wavefunction and potential#
    #psi=coherent_state(1)
    psi=energy_eigenstate(5)

    v_n=harmonic_potential(x)

    #Begin evolution of system#
    qho = System(x, psi(x), v_n)
    
    cor_func_t=[]
    for t in np.arange(Nt_steps):
        qho.time_evolve(dt, 1)
        psi=qho._get_psi_x()
        psi0=qho._get_initial_psi()
        temp_val=np.trapz(np.conjugate(psi0)*psi, x=qho.x)
        cor_func_t.append(temp_val)
    cor_func_t=np.array(cor_func_t)

    f,Sxx=signal.periodogram(cor_func_t, 100, window='hann', scaling='density')


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #Plotting#
    fig, ax = plt.subplots()
    #dx/sqrt(2pi) * psi(x_n)* exp(-ik0x_n) <--> \tilde(psi)(k_m)exp(-im*x0*dk)
    ax.plot(f, np.sqrt(Sxx))
    ax.set_xlim([-5, 5])

    ax.set_title("n=5 Excited State Power Spectrum")
    ax.set_xlabel(r"Energy $\hbar \omega$", fontsize=16)
    ax.set_ylabel(r"Spectral Power", fontsize=16)

    plt.show()

    fig.savefig("n5_excited_state_spectrum.png")
if __name__=="__main__":
    main()

