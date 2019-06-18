# Simulating quantum systems using a spectral method 

This repository contains code to simulate an arbitary 1D quantum system by specifying a potential and 
a given wavefunction. This specification is done via configuration files as detailed below. The simulation
implements a split operator spectral method that is O(t^2).

## Downloading

To install, simply clone this repository using

```
git clone https://github.com/gplynch619/quantum.git
```

## Running

Currently, only the 1D simulator is working. This simulator is `main.py` in the `python_files` folder. In order to run it, use
```
python main.py <path/to/config>
```

This will produce a numpy file containing the energy spectrum obtained after running the simulation.

## Configuration

The simulator loads its paramters from a user defined configuration file. There is a configuration file included that specifies a simulation of the quantum harmonic oscillator. The configuration file has the following formation

```yaml
---
mass: &mass         #mass of the particle 
hbar: &hbar         #Reduced planck's constant
omega: &omega       #angular frequency

T: 	                #Total duration of simulation
ntime: 	            #Total number of time steps

N:                  #Total number of spatial samples
xstart:             #Starting x value
xstop_or_dx:        #if endpoint_mode, then this is xstop. If not, this is dx.
endpoint_mode:      #If endpoint_mode is true, then the grid initialization a grid between xstart and xstop with N grid points. If it is false, then it creates a grid starting at xstart, with with grid spacing of dx for N grid points

output:      		#If this is specified, the simulation saves to the specified file
potential:
  type:             #This is the name of the potential function. There should be a corresponding function defined in potentials.py
  params:           # these are the parameters passed to the potential function. They should be properly hooked in in potentials.py
    mass: *mass
    omega: *omega
    hbar: *hbar
wavefunction:
  type:             #This is the name of the wavefunction function. There should be a corresponding function defined in wavefunctions.py
  params:           #these are the parameters passed to the potential function. They should be properly hooked in in potentials.py
    level: 0
    mass: *mass
    omega: *omega
    hbar: *hbar
```

For a full example, please see the included `qho_config.yaml` file. New wavefunctions and potentials can be defined, but they must be added to wavefunction.py and potentials.py respectively.

##Timing 

This algorithm is O(t^2) as evidenced by the following:

![Alt text](timing/fig1.png?raw=true "Title")

