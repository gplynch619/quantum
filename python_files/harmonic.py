'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This file contains potential and wave function information for the quantum harmonic oscillator
'''
import math
import numpy as np

from scipy.special import hermite

def harmonic_potential(params):
    
    def v(x):
        m=params['mass']
        omega=params['omega']
        return 0.5*m*(omega**2)*x*x #1/2m*w^2*x^2 but ignoring constants for n.0001avefunction states##
    return v

def energy_eigenstate(params): #Factory function to generate psi
    ''' This function is used to generate the energy eigenstates of the QHO.
    Parameters
    -------
    energy_level : an integer indicating which excited state should be generated.
    0 is the ground state and n is the nth excided state

    Returns
    -------
    psi(x) : A function that when evaluted gives the value of the indicated state at the position.
    '''
    n=params['level']
    m=params['mass']
    omega=params['omega']
    hbar=params['hbar']

    def psi(x):
        return ((m*omega/(np.pi*hbar))**0.25)*(1.0/np.sqrt((2**n)*math.factorial(n)))*np.exp((-m*omega*x**2)/2*hbar)*hermite(n)(np.sqrt(m*omega/hbar)*x)
    
    return psi

def coherent_state(params):
    a=params['a'] 
    def psi(x):
        return (1/np.pi**0.25)*np.exp(-(np.sqrt(0.5)*x-a)**2 + 0.5*(a**2- np.abs(a)**2))
    return psi
