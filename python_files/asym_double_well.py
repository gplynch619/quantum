'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This file defines the potential for the asymmetric double well, with constants chosen
to match Feit et al. 1982 (J. Comp Phys 47, 412-433)
'''
import math
import numpy as np
import constants


def asym_double_well_potential():
    def v(x):
        k0=-132.7074997
        k2=7.0
        k3=0.5
        k4=1.0
        return k0 - k2*x**2 + k3*x**3 + k4*x**4
    return v

def gaussian_super(a=constants.A, sigma=constants.SIGMA): #Factory function to generate psi
    
    m=constants.M/2.0
    hbar=constants.HBAR
    def psi(x):
        return np.exp((-(x-a)**2)/(2.0*sigma**2)) + np.exp((-(x+a)**2)/(2.0*sigma**2))
    
    return psi
