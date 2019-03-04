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

from scipy.special import hermite

class System(object):
    
    '''
    This class is the basic object which represents a quantum system. It consists of an 
    initial wavefunction, spatial domain, and potential.
    '''

    def __init__(self, x, psi_x_initial, v_x, hbar=1):
        
        self.N=x.shape[0]
        self.dx=x[1]-x[0]
        self.dp=2*np.pi/(self.N*self.dx)
        
        self.p0=-0.5*self.N*self.dp
        self.p=self.p0 + self.dp*np.arange(self.N)

        self._set_psi_x(psi_x_initial)
        self.compute_psi_p()
        
        self.v_x=v_x
        self.x=x
    #Setter and getter functions for psi x and p
    def _set_psi_x(self, psi_x):
        self._psi_x=psi_x*np.exp(-1.j*self.p[0]*self.x)*self.dx/np.sqrt(2*np.pi)

    def _get_psi_x(self):
        return self._psi_x*np.exp(1.j*self.p[0]*self.x)*np.sqrt(2*np.pi)/self.dx

    def _set_psi_p(self, psi_p):
        self._psi_p = psi_p*np.exp(1.j*self.x[0]*self.dk*np.arange(self.N))

    def _get_psi_p(self):
        return self._psi_p*np.exp(-1.jself.x[0]*self.dk*np.arange(self.N))
    #####

    def compute_psi_p(self):
        self._psi_p=np.fft.fft(self._psi_x)

    def compute_psi_x(self):
        self._psi_x=np.fft.ifft(self._psi_p)

    def time_evolve(self, dt, Nt_step):
        self.dt=dt

        self._psi_x *= np.exp(-1.j*0.5*dt*self.v_x)
        
        for i in xrange(Nt_step):
            self.compute_psi_p()
            self._psi_p *= np.exp(-1.j*dt*0.5*self.p*self.p)
            self.compute_psi_x()
            self._psi_x *= np.exp(-1.j*dt*self.v_x)

        self.compute_psi_p()
        self._psi_p*=np.exp(-1.j*0.5*dt*self.p*self.p)

        self.comput_psi_x() 
        self._psi_x *= np.exp(-1.j*0.5*dt*self.v_x)

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

def harmonic_potential(x):
    return 0.5*x*x #1/2m*w^2*x^2 but ignoring constants for now

def main():
    ##Defining Parameters##
    dt=.05     #arbitrary 'small' dt 
    hbar=1      #Reduced Planck                      
    N=2**11       #number of spatial samples
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

    harmonic_osc = System()

    #Time Evo#
    
    #dx/sqrt(2pi) * psi(x_n)* exp(-ik0x_n) <--> \tilde(psi)(k_m)exp(-im*x0*dk)
    ####

    #operator_exp(x)(t/2)F^-1[operator_exp(p)F(operator_exp(x)(t/2)*psi(x))]


    fig, ax = plt.subplots()
    ax.plot(x, psi_x_periodic.real, x, psi_x_periodic.imag)

    #ax.plot(x, intermed_psi.real, x, intermed_psi.imag)

    #ax.grid(True)

    plt.show()

if __name__=="__main__":
    main()

