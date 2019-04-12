'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This file defines the potential for the asymmetric double well, with constants chosen
to match Feit et al. 1982 (J. Comp Phys 47, 412-433)
'''
import math
import numpy as np
import constants


def asym_double_well_potential(params):
    def v(x):
        k0=params['k0']
        k2=params['k2']
        k3=params['k3']
        k4=params['k4']
        return k0 - k2*x**2 + k3*x**3 + k4*x**4
    return v

def gaussian_super(a=constants.A, sigma=constants.SIGMA): #Factory function to generate psi
    a=params['a']
    sigma=params['sigma']
    m=params['mass']
    hbar=params['hbar']
    def psi(x):
        return np.exp((-(x-a)**2)/(2.0*sigma**2)) + np.exp((-(x+a)**2)/(2.0*sigma**2))
    
    return psi
