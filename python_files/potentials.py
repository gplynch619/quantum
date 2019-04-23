import numpy as np

def asym_double_well_potential(params):
    def v(x):
        k0=params['k0']
        k2=params['k2']
        k3=params['k3']
        k4=params['k4']
        return k0 - k2*x**2 + k3*x**3 + k4*x**4
    return v

def harmonic_potential(params):
    
    def v(x):
        m=params['mass']
        omega=params['omega']
        return 0.5*m*(omega**2)*x*x #1/2m*w^2*x^2 but ignoring constants for n.0001avefunction states##
    return v

def nonlinear_potential(params):
    def v(psi):
        beta=params['beta']
        return beta*np.conj(psi)*psi
    return v
