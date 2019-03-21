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
import constants
from matplotlib import animation
from quantum_system import *

from scipy.special import hermite

def harmonic_potential(x):
    m=constants.M
    omega=constants.OMEGA
    return 0.5*m*omega*x*x #1/2m*w^2*x^2 but ignoring constants for now

##Wavefunction states##
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
    m=constants.M
    omega=constants.OMEGA
    hbar=constants.HBAR
    def psi(x):
        return ((m*omega/(np.pi*hbar))**0.25)*(1.0/np.sqrt((2**n)*math.factorial(n)))*np.exp((-m*omega*x**2)/(2*hbar))*hermite(n)(np.sqrt(m*omega/hbar)*x)
    
    return psi

def coherent_state(a):
    
    def psi(x):
        return (1/np.pi**0.25)*np.exp(-(np.sqrt(0.5)*x-a)**2 + 0.5*(a**2- np.abs(a)**2))
    return psi

def main():
    ##Defining Parameters##

    m=constants.M
    omega=constants.OMEGA
    hbar=constants.HBAR

    sampling=100.0      #arbitrary 'small' dt 
    nsamples=1256
    dt=1.0/sampling
    N=2**9     #number of spatial samples
    T=nsamples*dt

    frames = int(3*T)
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
    #psi=coherent_state(1)
    psi=energy_eigenstate(0)

    v_n=harmonic_potential(x)

    #Begin evolution of system#
    harmonic_osc = System(x, psi(x), v_n)

    #Uncomment the following lines if you just want to evolve to some endpoint
    #harmonic_osc.time_evolve(dt, Nt_steps)
    #final = harmonic_osc._get_psi_x()
    
    #fig1, ax=plt.subplots()
    #ax.plot(harmonic_osc.x, final)
    ####

    #dx/sqrt(2pi) * psi(x_n)* exp(-ik0x_n) <--> \tilde(psi)(k_m)exp(-im*x0*dk)
    
    ##Animation##
    #Setting up plot
    fig, axes = plt.subplots(1,2)
    
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

    state='a=1 Coherent State of QHO'

    axes[0].set_title(r"$\psi$, {}".format(state), fontsize=16)
    axes[1].set_title(r"$|\psi|^2$, {}".format(state), fontsize=16)

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
        harmonic_osc.time_evolve(dt, 33)
        psi_real_line.set_data(harmonic_osc.x, harmonic_osc._get_psi_x().real)
        psi_imag_line.set_data(harmonic_osc.x, harmonic_osc._get_psi_x().imag)
        psi_square_line.set_data(harmonic_osc.x, np.abs(harmonic_osc._get_psi_x())**2)
        text1.set_text('t={}'.format(harmonic_osc.t))
        text2.set_text('t={}'.format(harmonic_osc.t))
        return psi_real_line, psi_imag_line, psi_square_line
        print(i)

    axes[0].grid(True)
    axes[1].grid(True)
    fps=15.0
    mult=(1.0/fps)*1000
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=mult, blit=False)
        
    anim.save('test.gif', dpi=80, writer='imagemagick')

    plt.show()

if __name__=="__main__":
    main()

