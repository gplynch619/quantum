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

def harmonic_potential(x):
    return 0.5*x*x #1/2m*w^2*x^2 but ignoring constants for now

##Wavefunction states##
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
    
    dt=.01      #arbitrary 'small' dt 
    hbar=1      #Reduced Planck                      
    N=2**11     #number of spatial samples
    Nt_steps=50
    t_max = 12
    frames = int(t_max/float(Nt_steps*dt))
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
    psi=energy_eigenstate(0)

    v_n=harmonic_potential(x)

    #Begin evolution of system#
    harmonic_osc = System(x, psi(x), v_n)
    harmonic_osc.time_evolve(dt, Nt_steps)
    final = harmonic_osc._get_psi_x()
    
    #Plotting#
    fig, ax = plt.subplots()
    ax.plot(harmonic_osc.x, final)


    #dx/sqrt(2pi) * psi(x_n)* exp(-ik0x_n) <--> \tilde(psi)(k_m)exp(-im*x0*dk)
    
    plt.show()

if __name__=="__main__":
    main()

