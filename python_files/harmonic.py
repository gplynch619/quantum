'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This program simulates the quantum dynamics of certain quantum harmonic oscillator (QHO) 
states via an implementation of the spectral method (see Feit et. al. 1982). It uses the
built-in FFT that comes with numpy for its Fourier Transforms. This is currently all done
in one dimension but the generalization to 3 dimensions should be straightforward, with
special attention paid to parallelization. Units are SI.

'''
import math
import numpy as np
import matplotlib.pyplot as plt
import constants
from quantum_system import *

from scipy.special import hermite
from scipy import signal
from scipy import optimize

def harmonic_potential(x):
    m=constants.M
    omega=constants.OMEGA
    return 0.5*m*(omega**2)*x*x #1/2m*w^2*x^2 but ignoring constants for n.0001avefunction states##
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
        return ((m*omega/(np.pi*hbar))**0.25)*(1.0/np.sqrt((2**n)*math.factorial(n)))*np.exp((-m*omega*x**2)/2*hbar)*hermite(n)(np.sqrt(m*omega/hbar)*x)
    
    return psi

def coherent_state(a):
    
    def psi(x):
        return (1/np.pi**0.25)*np.exp(-(np.sqrt(0.5)*x-a)**2 + 0.5*(a**2- np.abs(a)**2))
    return psi

def main():
    ##Defining Parameters##
    
    m=constants.M
    omega=constants.OMEGA
    hbar=constants.HBAR      #Reduced Planck                      
    
    sampling=100.0 #how many time steps per sec
    nsamples=16384
    dt=1.0/sampling
    N=2**9     #number of spatial samples
    T=nsamples*dt
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
    qho = System(x, psi(x), v_n)
    time_steps=np.arange(0.0, T, dt) 
    cor_func_t=[]
    for step in time_steps:
        qho.time_evolve(dt, 1)
        psi=qho._get_psi_x()
        psi0=qho._get_initial_psi()
        temp_val=np.trapz(np.conjugate(psi0)*psi, x=qho.x)
        cor_func_t.append(temp_val)
    cor_func_t=np.array(cor_func_t)
    
    def test_func(t, a, w, d):
        return a*np.sin(w*t+d)

    params, params_covariance = optimize.curve_fit(test_func, time_steps, cor_func_t.real)

    print("Corr func fit to a*sin(wt+d):")
    print("Best fit a: {}".format(params[0]))
    print("Best fit omega: {}".format(params[1]))
    print("Best fit phase: {}".format(params[2]))

#    signal_length=len(cor_func_t)
#    ks=np.arange(signal_length)
#    T = signal_length/sampling
#    E = ks/T
#    E=E[range(signal_length/2)]
    #E*=-1

#    energy_spec=np.fft.fft(cor_func_t)/signal_length
#    energy_spec=energy_spec[range(signal_length/2)]
    
    f, pxx = signal.periodogram(cor_func_t, sampling, window='hann', scaling='density')
    
    temp = np.split(f, 2) #originally in (0, ..., n/2, -n/2, ... 1)
    f=np.append(temp[1], temp[0])
    temp=np.split(pxx, 2) # same as above
    pxx=np.append(temp[1], temp[0])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(2,1)
    #dx/sqrt(2pi) * psi(x_n)* exp(-ik0x_n) <--> \tilde(psi)(k_m)exp(-im*x0*dk)

    ax[0].plot(time_steps, cor_func_t.real)
    ax[0].set_title("Correlation Function")
    ax[0].set_xlim([0, 1])
    ax[0].set_xlabel(r"Time $t$", fontsize=16)
    ax[0].set_ylabel(r"Amplitude", fontsize=16)
    
#    ax[1].plot(E, np.abs(energy_spec))
    ax[1].plot(f*2*np.pi, pxx, 'r')
    ax[1].set_xlabel(r"Energy ($\hbar \omega$)")
    ax[1].set_ylabel(r"Spectral power")
    ax[1].set_xlim([-5, 0])
    plt.show()

    #fig.savefig("example.png")
if __name__=="__main__":
    main()

