'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC

This program simulates the quantum dynamics of a quantum system defined by a potential
and initial wavefunction. It is an implementation of the spectral method (see Feit et. al. 1982). It uses the
built-in FFT that comes with numpy for its Fourier Transforms. 

'''
import os
import sys
import math
import yaml
import time
import datetime
import subprocess
import numpy as np
from quantum_system import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from importlib import import_module
from main import load_config, write_log

def x_conv_setup(config):
    ##Set up time steps##
    T=config['T']
    ntime=config['ntime']
    dt=float(T)/ntime

    ##Set up grid##
    N_str=config['N']
    N=[int(num) for num in N_str.split(" ")]
    xstart=config['xstart']
    xstop_or_dx=config['xstop_or_dx']
    grid_flag=config['endpoint_mode']
    x_arr=[]
    p_arr=[]
    for ng in N:
        x,p=initialize_grid(xstart, ng, xstop_or_dx, grid_flag)
        x_arr.append(x)
        p_arr.append(p)

    ##Set up wavefunction##
    psi_library=import_module('wavefunctions')
    psi=getattr(psi_library, config['wavefunction']['type'])(config['wavefunction']['params'])

    v_library=import_module('potentials')
    v=getattr(v_library, config['potential']['type'])(config['potential']['params'])

    return T,ntime,dt,x_arr,p_arr,psi,v

def find_nearest(arr, value):
    idx=(np.abs(arr-value)).argmin()
    return arr[idx]

def main():
    total_start=time.time()    
    
    outdir="outputs/"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    config=load_config(sys.argv[1])
    T,ntime,dt,xarr,parr,psi,v = x_conv_setup(config)
    sim_start=time.time()
    outdir='outputs/' 
    dxs=[]
    wfunc_rmse=[]
    for grid in xarr:
        try:
            qs = System(grid, psi, v, nonlinear=config['nonlinear'])
        except:   
            qs = System(grid, psi, v) 
        ts=np.fft.fftfreq(ntime, dt)
        ts=np.fft.fftshift(ts)
        cor_func_t=[]
        qs.time_evolve(dt, ntime, config)
        
        psi_cur=np.abs(qs._get_psi_x())**2
        psi0=np.abs(qs._get_initial_psi())**2
        mse=np.mean((psi_cur-psi0)**2)
        rmse=np.sqrt(mse)
        
        wfunc_rmse.append(rmse)
        dxs.append(qs.dx)
        del qs 
    sim_end=time.time() 
    
    wfunc_rmse=np.array(wfunc_rmse)
    dxs=np.array([dxs])
    Narr=[len(x) for x in xarr]
    print("NARRRRR")
    print(*Narr)
    print(xarr[0])
    sim_time=sim_end-sim_start
    ###LOG OUTPUT### 
    
    output_files=[]
    now = datetime.datetime.now()
    
    if config['save_file']:
        cfg_name=os.path.split(sys.argv[1])[-1]
        cfg_name=cfg_name.split(".")[0]
        try:
            filename=config['save_file_name']+"_"+now.strftime("%Y-%m-%d")+'.npz' 
        except:
            filename=cfg_name+"_"+now.strftime("%Y-%m-%d")+'.npz' 
        np.savez(outdir+filename, rmse=wfunc_rmse, ng=Narr, dxs=dxs) 
        output_files.append(filename)

    rmse_info="rmse estimates are {}".format(wfunc_rmse)
    header="==============================================\nRECORD CREATED {0}\n".format(now.strftime("%Y-%m-%d %H:%M"))
    summary="Simulation potential: {0}\nSimulation wavefunction: {1}".format(config['potential']['type'], config['wavefunction']['type'])
    output_info="Simulation produced outputs: {}".format(" ".join(f for f in output_files))
    cfg_blurb="The following configuration was used: {}".format(os.path.split(sys.argv[1])[-1])

    total_end=time.time()
    total_time=total_end-total_start
    timing_info="Simulation time: {0}\nTotal time {1}".format(sim_time, total_time)
    
    strings=[header, timing_info, summary, rmse_info, output_info, cfg_blurb]
   
    [print(s) for s in strings]

    write_log(strings, config)
    
    sys.exit()
if __name__=="__main__":
    main()

