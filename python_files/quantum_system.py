'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This class is the basic object which represents a quantum system. It consists of an 
initial wavefunction, spatial domain, and potential.

'''

import math
import numpy as np

class System(object):
    
    def __init__(self, x, psi, v, nonlinear=False, bc='periodic'):
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
        
        if bc=='periodic':
            self.x=Periodic_Grid(x)
        else:
            self.x=x
            
        self.t=0

        self.N=self.x.shape[0]
        self.dx=self.x[1]-self.x[0]
        
        self.dp=2*np.pi/(self.N*self.dx)
        self.p0=-0.5*self.N*self.dp
        self.p=self.p0 + self.dp*np.arange(self.N)
        
        self._set_psi_x(psi(self.x))
        self._set_initial_psi(psi(self.x))
        self.compute_psi_p()

        if self.nonlinear:
            self.v_x=v(np.asarray(self._get_psi_x()))
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
        self.v_x=self.v(np.asarray(self._get_psi_x()))

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

class Periodic_Grid(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.dtype=input_array.dtype
        obj.grid_size = input_array.size
        return obj

    def __getitem__(self, index):
        if isinstance(index, slice):
            start=None
            stop=None
            print(index) 
            
            if index.start is None and index.stop is not None:
                start=0
                stop=index.stop
            elif index.start is not None and index.stop is None :
                start=index.start
                stop=super().size
            elif index.start is None and index.stop is None:
                return np.asarray(super().__getitem__(index))
            else:
                start=index.start
                stop=index.stop
            
            indices=list(range(start, stop))
            size=len(indices)
            a=np.empty(size, dtype=self.dtype)
            print("Size: {0} indices: {1}".format(size, indices))
            for i, ind in enumerate(indices): 
                idx=self.wrapped_index(ind)
                a[i]=self[idx]
            return a

        elif isinstance(index, tuple):
            idx=index[0]
            idx=self.wrapped_index(idx)
            return super().__getitem__(idx)
        elif isinstance(index, int):
            index = self.wrapped_index(index)
            return super().__getitem__(index)
        else:
            print("RAISING EXCEPTION")
            raise ValueError
        
    def __setitem__(self, index, item):
        index = self.wrapped_index(index)
        return super().__setitem__(index, item)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.grid_size = getattr(obj, 'grid_size', obj.size)
        pass

    def wrapped_index(self, index):
        if type(index) is tuple:
            index=index[0] #only support for 1D!
            mod_index = index%self.grid_size
            return mod_index
        elif type(index) is int:
            mod_index = index%self.grid_size
            return mod_index
        else:
           return index 

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
