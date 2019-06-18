'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This class is the basic object which represents a quantum system. It consists of an 
initial wavefunction, spatial domain, and potential.

'''

import math
import numpy as np

class System(object):
    
    def __init__(self, x, psi, v, nonlinear=False):
        '''
        Initializer for quantum system
        Parameters
        -------
        x : A numpy array containing evenly spaced samples of the wavefunction domain. x[0]=x0
        and x[-1]=xf. The number of samples N is computed from the size of this array

        psi_x_initial : An array of size N containing the initial values for the wavefunction
        to be evolved

        v_x : An array of size N containing the values for the system potential

        nonlinear : A boolean specifying whether the system is linear or nonlinear
        '''
        self.nonlinear=nonlinear
        self.v=v

        self.N=x.shape[0]
        self.dx=x[1]-x[0]
        self.dp=2*np.pi/(self.N*self.dx)
        
        self.p0=-0.5*self.N*self.dp
        self.p=self.p0 + self.dp*np.arange(self.N)
        
        self.x=x
        self.t=0
        
        self._set_psi_x(psi(x))
        self._set_initial_psi(psi(x))
        self.compute_psi_p()

        if self.nonlinear:
            self.v_x=v(self._get_psi_x())
        else:
            self.v_x=v(self.x)

     #Setter and getter functions for psi x and p, and v
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
    def update_v_x(self):
        self.v_x=self.v(self._get_psi_x())

    def is_nonlinear(self):
        return self.nonlinear

    def compute_psi_p(self):
        self._psi_p=np.fft.fft(self._psi_x)

    def compute_psi_x(self):
        self._psi_x=np.fft.ifft(self._psi_p)

    def time_evolve(self, dt, Nt, params): #this is where the magic happens
        self.dt=dt
        hbar=params['hbar']
        m=params['mass']

        #the negative comes from fourier transforming the (p <-> -ih\nabla)
        #so \nabla^2 <-> -p^2/h^2

        for i in np.arange(Nt):
            self._psi_p *= np.exp(-1.j*dt*0.25*self.p*self.p/(hbar*m))
            self.compute_psi_x()
            self._psi_x *= np.exp(-1.j*dt*self.v_x)
            self.compute_psi_p()
            self._psi_p *= np.exp(-1.j*dt*0.25*self.p*self.p/(hbar*m))
            if self.is_nonlinear():
                self.compute_psi_x
                self.update_v_x()

        self.t += dt*Nt

def initialize_grid(xstart,  ng, third, endpoint_mode=True):
    #this initializes a grid between xstart and xstop with ng points
    #the momentum grid is calculated from the dual to the position grid
    #this grid is then used to initialize the system (in System.__init__())
    if endpoint_mode:
        x0=xstart
        xf=third
        dx=(xf-x0)/ng
        x=[x0+n*dx for n in range(ng)]
        x=np.array(x)
    else:
        x0=xstart
        dx=third
        x=[x0+n*dx for n in range(ng)]
        x=np.array(x)

    p0=-0.5*ng*dx
    dp=2*np.pi/(ng*dx)
    p=[p0+m*dp for m in range(ng)]
    p=np.array(p)

    return x,p