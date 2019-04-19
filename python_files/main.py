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

def load_config(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def setup(config):
    ##Set up time steps##
    T=config['T']
    ntime=config['ntime']
    dt=float(T)/ntime

    ##Set up grid##
    N=config['N']
    xstart=config['xstart']
    xstop_or_dx=config['xstop_or_dx']
    grid_flag=config['endpoint_mode']

    x,p = initialize_grid(-20, N, 20.0, endpoint_mode=grid_flag)

    ##Set up wavefunction##
    psi_library=import_module('wavefunctions')
    psi=getattr(psi_library, config['wavefunction']['type'])(config['wavefunction']['params'])

    v_library=import_module('potentials')
    v=getattr(v_library, config['potential']['type'])(config['potential']['params'])

    return T,ntime,dt,x,p,psi,v

def write_log(strings, config):
    
    logfile='sim.log'
    if 'logfile' in config:
        logfile=config['logfile']
    for string in strings:
        subprocess.call("echo -e '{0}' >> {1}".format(string, logfile), shell=True)
    subprocess.call("awk 'NF' {0} >> {1}".format(sys.argv[1], logfile), shell=True)

def main():
    total_start=time.time()    
    config=load_config(sys.argv[1])
    T,ntime,dt,x,p,psi,v = setup(config)

    outdir="outputs/"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    try:
        qs = System(x, psi, v, nonlinear=config['nonlinear'])
    except:   
        qs = System(x, psi, v) 
    ts=np.fft.fftfreq(ntime, dt)
    ts=np.fft.fftshift(ts)
    cor_func_t=[]
    sim_start=time.time()
    for step in ts:
        qs.time_evolve(dt, 1, config)
        psi=qs._get_psi_x()
        psi0=qs._get_initial_psi()
        temp_val=np.trapz(np.conjugate(psi0)*psi, x=qs.x)
        cor_func_t.append(temp_val)
    cor_func_t=np.array(cor_func_t)

    energy_spec=np.fft.fft(cor_func_t)/ntime
    energy_spec=np.fft.fftshift(energy_spec)    
    peaks,_=find_peaks(energy_spec)
    peaks_t=2*np.pi*ts[peaks]
    sim_end=time.time()


    fig,ax=plt.subplots()
    
    ax.plot(x, np.abs(qs._get_psi_x())**2)
   
    #SAVING
    output_files=[]
    
    now = datetime.datetime.now()
    try:
        filename=config['plot_file']+'.png'
        plt.savefig(filename)
        output_files.append(filename)
    except:
        plt.show()
    
    try:
        filename=config['save_file']+'_'+now.strftime("%Y-%m-%d")+'.npz'
        np.savez(outdir+filename, energy_spec=energy_spec, ts=ts)
        output_files.append(filename)
    except Exception as e:
        print("No file saved")
        print(e)
        pass

    total_end=time.time()
    ###LOG OUTPUT### 
    sim_time=sim_end-sim_start
    total_time=total_end-total_start
    
    timing_info="Simulation time: {0}\nTotal time {1}".format(sim_time, total_time)
    peak_info="Peaks at {}".format(peaks_t)
    header="==============================================\nRECORD CREATED {0}\n".format(now.strftime("%Y-%m-%d %H:%M"))
    summary="Simulation potential: {0}\nSimulation wavefunction: {1}".format(config['potential']['type'], config['wavefunction']['type'])
    output_info="Simulation produced outputs: {}".format(" ".join(f for f in output_files))
    cfg_blurb="The following configuration was used: {}".format(os.path.split(sys.argv[1])[-1])
   
    strings=[header, summary, timing_info, output_info, cfg_blurb]
    write_log(strings, config)
    [print(s) for s in strings]

    sys.exit()
if __name__=="__main__":
    main()

