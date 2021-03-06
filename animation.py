'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC
'''
import os
import math
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from quantum_system import *
from main import load_config, setup

def main():
    
    cfg_name=os.path.split(sys.argv[1])[-1]
    cfg_name=cfg_name.split(".")[0]
    config=load_config(sys.argv[1])
    T,ntime,dt,x,p,psi,v=setup(config)

    frames = int(3*T)
    
    qs = System(x, psi, v, nonlinear=config['nonlinear'])

    ##Animation##
    #Setting up plot
    fig, axes = plt.subplots(1,2)
    
#    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    psi_real_line, = axes[0].plot([], [], lw=2)
    psi_imag_line, = axes[0].plot([], [], lw=2, color='orange')
    psi_square_line, = axes[1].plot([], [], lw=2)

    for ax in axes:
        ax.set_xlim([qs.x[0], qs.x[-1]])
        ax.set_ylim([-1, 1])
   
    text1 = axes[0].text(.15, 0.9, "", fontsize=16 ,transform=axes[0].transAxes)
    text2 = axes[1].text(.15, 0.9, "", fontsize=16, transform=axes[1].transAxes)
    if 'base_file_name' in config:
        state_name=config['base_file_name']
    else:
        state_name=cfg_name

    axes[0].set_title(r"$\psi$, {}".format(state_name), fontsize=16)
    axes[1].set_title(r"$|\psi|^2$, {}".format(state_name), fontsize=16)

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
        qs.time_evolve(dt, 1, config)
        psi_real_line.set_data(qs.x, qs._get_psi_x().real)
        psi_imag_line.set_data(qs.x, qs._get_psi_x().imag)
        psi_square_line.set_data(qs.x, np.abs(qs._get_psi_x())**2)
        text1.set_text('t={:3.2f}'.format(qs.t))
        text2.set_text('t={:3.2f}'.format(qs.t))
        return psi_real_line, psi_imag_line, psi_square_line
        print(i)

    axes[0].grid(True)
    axes[1].grid(True)
    fps=15.0
    mult=(1.0/fps)*1000
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=mult, blit=False)
    
    now = datetime.datetime.now()
    if config['save']:    
        if 'base_file_name' in config:
            try:
                filename=sys.argv[2]+"_"+config['base_file_name']+"_"+now.strftime("%Y-%m-%d")+'.gif' 
            except:
                filename=config['base_file_name']+"_"+now.strftime("%Y-%m-%d")+'.gif'
        else:
            try:
                filename=sys.argv[2]+cfg_name+"_"+now.strftime("%Y-%m-%d")+'.gif' 
            except:
                filename=cfg_name+"_"+now.strftime("%Y-%m-%d")+'.gif'
        anim.save(filename, dpi=80, writer='imagemagick')
        print("Animation saved as {}".format(filename))
        
    plt.show()

if __name__=="__main__":
    main()

