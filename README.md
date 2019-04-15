# Schrodinger Formalism Structure Formation

This repository contains code to implement structure formation simulations using the Schrodinger formalism (see Widrow & Kaiser 1993). Currently, there are two main lines of development: a 1D simulator written in Python to be used for testing the main features of the algorithm and implementation. It will likely be removed or mothballed once the full simulation code is written.

In addition to the Python simulator, there is a main simulator written in C++ using SWFFT and the HACC initializer. This has yet to be developed. 

## Installation

To install, simply clone this repository using

```
git clone https://github.com/gplynch619/quantum.git
```

Running make in the main directory will compile the C++ portion using the included makefile. It may be necessary to modify the makefile to get it to compile on your system. The Python portion needs no further set up.

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
For a full example, please see the included `qho_config.yaml` file.

