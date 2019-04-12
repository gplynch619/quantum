'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC
'''

import os
import math
import numpy as np
import constants

from scipy.signal import find_peaks
from harmonic import *
from quantum_system import *
from mpi4py import  MPI

def main():
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    m=constants.M
    omega=constants.OMEGA
    hbar=constants.HBAR      #Reduced Planck                      

    N=2**9     #number of spatial samples
    x,p = initialize_grid(-10, N, 10.0, endpoint_mode=True)
    psi=energy_eigenstate(0)           #from harmonic.py
    v=harmonic_potential()         #from harmonic.py
    v_n=v(x)
    qs = System(x, psi(x),v_n)
    ntimes_array=None 
    dts=[]
    peaks=[]
    steps=[]
    T=4000
    if rank==0:
        ntimes_array=np.arange(start=20, stop=81940, step=10, dtype='i')
        data_size=ntimes_array.size
        if not os.path.exists("MPI_{}T".format(T)):
            os.mkdir("MPI_{}T".format(T))
    else:
        data_size=None

    data_size=comm.bcast(data_size, root=0)
    local_size = int(data_size/size)
    ntimes_local=np.empty(data_size, dtype='i')
    comm.Scatter(ntimes_array, ntimes_local, root=0)
    ntimes_local=np.resize(ntimes_local, local_size)
    print("proc {0} has ntimes_local:\t{1}".format(rank, ntimes_local))

    for ntime in ntimes_local: 
        dt= float(T)/ntime
        ts=np.fft.fftfreq(ntime, dt)
        ts=np.fft.fftshift(ts)
        cor_func_t=[]
        for step in ts:
            qs.time_evolve(dt, 1)
            psi=qs._get_psi_x()
            psi0=qs._get_initial_psi()
            temp_val=np.trapz(np.conjugate(psi0)*psi, x=qs.x)
            cor_func_t.append(temp_val)
        cor_func_t=np.array(cor_func_t)

        energy_spec=np.fft.fft(cor_func_t)/ntime
        energy_spec=np.fft.fftshift(energy_spec)    

        peaks_indx,_=find_peaks(np.abs(energy_spec))
        peaks_e=2*np.pi*ts[peaks_indx]
        peaks.append(peaks_e[-1])
        dts.append(dt)
        steps.append(ntime)
        print("Rank {0} completed ntime {1}".format(rank, ntime))

    dts=np.array(dts)
    peaks=np.array(peaks)
    steps=np.array(steps)
    
    if rank==0:
        global_dts=np.empty(data_size, dtype='f')
        global_peaks=np.empty(data_size, dtype='f')
        global_steps=np.empty(data_size, dtype='i')

    comm.Gather(dts, global_dts, root=0)
    comm.Gather(peaks, global_peaks, root=0)
    comm.Gather(steps, global_steps, root=0)
    if rank==0: 
        with open("MPI_{0}T/MPI_peaks.npz".format(T)) as filename:
            np.savez(filename, global_dts, global_steps, global_peaks)

if __name__=="__main__":
    main()

