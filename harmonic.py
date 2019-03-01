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
'''
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import hermite

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

def main():
    dt=.05 #arbitrary 'small' dt 
    psi = energy_eigenstate(0)
    x = np.linspace(-5.0, 5.0, 100)

    

    #Psi has been initialized, and to sample over the grid we do psi(x) 

    fig, ax = plt.subplots()

    
    
    #ax.grid(True)

    plt.show()

if __name__=="__main__":
    main()

