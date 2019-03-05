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
from matplotlib import animation

from scipy.special import hermite

class System(object):
    
    '''
    This class is the basic object which represents a quantum system. It consists of an 
    initial wavefunction, spatial domain, and potential.
    '''

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
        self.compute_psi_p()
        
     #Setter and getter functions for psi x and p
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
    
    dt=.01      #arbitrary 'small' dt 
    hbar=1      #Reduced Planck                      
    N=2**11     #number of spatial samples
    Nt_steps=50
    t_max = 12
    frames = int(t_max/float(Nt_steps*dt))
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
    psi=energy_eigenstate(1)

    v_n=harmonic_potential(x)

    #Begin evolution of system#
    harmonic_osc = System(x, psi(x), v_n)

    #Uncomment the following lines if you just want to evolve to some endpoint
    #harmonic_osc.time_evolve(dt, Nt_steps)
    #final = harmonic_osc._get_psi_x()
    ####

    
    #dx/sqrt(2pi) * psi(x_n)* exp(-ik0x_n) <--> \tilde(psi)(k_m)exp(-im*x0*dk)
    #operator_exp(x)(t/2)F^-1[operator_exp(p)F(operator_exp(x)(t/2)*psi(x))]
    
    ##Animation##
    #Setting up plot
    fig, axes = plt.subplots(1,2, figsize=(4,2))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    psi_real_line, = axes[0].plot([], [], lw=2)
    psi_imag_line, = axes[0].plot([], [], lw=2, color='orange')
    psi_square_line, = axes[1].plot([], [], lw=2)

    for ax in axes:
        ax.set_xlim([-6, 6])
        ax.set_ylim([-1, 1])
   
    axes[1].set_ylim([-.1, 1])
    text1 = axes[0].text(.15, 0.9, "", fontsize=16 ,transform=axes[0].transAxes)
    text2 = axes[1].text(.15, 0.9, "", fontsize=16, transform=axes[1].transAxes)

    axes[0].set_title(r"$\psi$, n=1 State of QHO", fontsize=16)
    axes[1].set_title(r"$|\psi|^2$, n=1 State of QHO", fontsize=16)

    axes[0].legend((psi_real_line, psi_imag_line), (r'$Re(\psi)$', r'$Im(\psi)$'))

    for ax in axes:
        ax.set_xlabel(r"Position $(\sqrt{\hbar/m \omega})$")

    def init():
        psi_real_line.set_data([],[])
        psi_imag_line.set_data([],[])
        psi_square_line.set_data([],[])
        text1.set_text("")
        text2.set_text("")
        return psi_real_line, psi_imag_line, psi_square_line, text1, text2

    def animate(i):
        harmonic_osc.time_evolve(dt, Nt_steps)
        psi_real_line.set_data(harmonic_osc.x, harmonic_osc._get_psi_x().real)
        psi_imag_line.set_data(harmonic_osc.x, harmonic_osc._get_psi_x().imag)
        psi_square_line.set_data(harmonic_osc.x, np.abs(harmonic_osc._get_psi_x())**2)
        text1.set_text('t={}'.format(harmonic_osc.t))
        text2.set_text('t={}'.format(harmonic_osc.t))
        return psi_real_line, psi_imag_line
    
    axes[0].grid(True)
    axes[1].grid(True)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=120, blit=False)
        
    anim.save('n1_state_animation.gif', dpi=80, writer='imagemagick')

    #ax.grid(True)

    plt.show()

if __name__=="__main__":
    main()

