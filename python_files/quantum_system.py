'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This class is the basic object which represents a quantum system. It consists of an 
initial wavefunction, spatial domain, and potential.

'''

import math
import numpy as np

class System(object):
    
    def __init__(self, x, psi_x_initial, v_x, hbar=1):
        '''
        Initializer for quantum system
        Parameters
        -------
        x : A numpy array containing evenly spaced samples of the wavefunction domain. x[0]=x0
        and x[-1]=xf. The number of samples N is computed from the size of this array

        psi_x_initial : An array of size N containing the initial values for the wavefunction
        to be evolved

        v_x : An array of size N containing the values for the system potential 
        '''
        self.N=x.shape[0]
        self.dx=x[1]-x[0]
        self.dp=2*np.pi/(self.N*self.dx)
        
        self.p0=-0.5*self.N*self.dp
        self.p=self.p0 + self.dp*np.arange(self.N)
        
        self.v_x=v_x
        self.x=x
        self.t=0
        
        self._set_psi_x(psi_x_initial)
        self._set_initial_psi(psi_x_initial)
        self.compute_psi_p()
        
     #Setter and getter functions for psi x and p
    def _set_initial_psi(self, psi0_x):
        self._psi0_x=psi0_x*np.exp(-1.j*self.p[0]*self.x)*self.dx/np.sqrt(2*np.pi)
    
    def _get_initial_psi(self):
        return self._psi0_x*np.exp(1.j*self.p[0]*self.x)*np.sqrt(2*np.pi)/self.dx

    def _set_psi_x(self, psi_x):
        self._psi_x=psi_x*np.exp(-1.j*self.p[0]*self.x)*self.dx/np.sqrt(2*np.pi)

    def _get_psi_x(self):
        return self._psi_x*np.exp(1.j*self.p[0]*self.x)*np.sqrt(2*np.pi)/self.dx

    def _set_psi_p(self, psi_p):
        self._psi_p = psi_p*np.exp(1.j*self.x[0]*self.dk*np.arange(self.N))

    def _get_psi_p(self):
        return self._psi_p*np.exp(-1.j*self.x[0]*self.dk*np.arange(self.N))
    #####

    def compute_psi_p(self):
        self._psi_p=np.fft.fft(self._psi_x)

    def compute_psi_x(self):
        self._psi_x=np.fft.ifft(self._psi_p)

    def time_evolve(self, dt, Nt_step): #this is where the magic happens
        self.dt=dt

        self._psi_x *= np.exp(-1.j*0.5*dt*self.v_x)
        
        for i in np.arange(Nt_step):
            self.compute_psi_p()
            self._psi_p *= np.exp(-1.j*dt*0.5*self.p*self.p)
            self.compute_psi_x()
            self._psi_x *= np.exp(-1.j*dt*self.v_x)

        self.compute_psi_p()
        self._psi_p*=np.exp(-1.j*0.5*dt*self.p*self.p)

        self.compute_psi_x() 
        self._psi_x *= np.exp(-1.j*0.5*dt*self.v_x)
        
        self.t += dt*Nt_step
