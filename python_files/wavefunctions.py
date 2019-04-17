import math
import numpy as np
from scipy.special import hermite

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

def gaussian_super(params): #Factory function to generate psi
    a=params['a']
    sigma=params['sigma']
    m=params['mass']
    hbar=params['hbar']
    def psi(x):
        return np.exp((-(x-a)**2)/(2.0*sigma**2)) + np.exp((-(x+a)**2)/(2.0*sigma**2))
    return psi

def bright_soliton(params):
    amp=params['amp']
    v=params['v']
    def psi(x):
        return amp/(np.cosh(amp*x))*np.exp(-1.j*v*x)
    return psi
