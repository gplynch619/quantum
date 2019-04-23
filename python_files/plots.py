import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    load_file=np.load(sys.argv[1])
    steps=load_file['steps']
    rmse=load_file['rmse']
    dts=load_file['dts']

    ratios=[]
    print(rmse)
    for i in np.arange(1, rmse.shape[0]-1): 
        num=rmse[i]
        den=rmse[i+1]
        ratios.append(np.log2(num/den))
    ratios=np.array(ratios)
    print("Covergence order: ",ratios)
    
    m0, b0=np.polyfit(np.log(dts), np.log(rmse), deg=1)
    yfit0=np.exp(m0*np.log(dts)+b0)
    
    m1, b1=np.polyfit(np.log(dts[5:]), np.log(rmse[5:]), deg=1)
    yfit1=np.exp(m1*np.log(dts)+b1)
    
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    fig, ax=plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Convergence order estimation, T=10")
    ax.set_xlabel("$\Delta$t", fontsize=14)
    ax.set_ylabel("$\delta$ (RMSE)", fontsize=14)
    
    ax.plot(dts, rmse, 'x')
    ax.plot(dts, yfit0, '--', label=r'Fit $p={:05.4f}$'.format(m0))
    ax.plot(dts, yfit1, '--', label=r'Fit (trimmed) $p={:05.4f}$'.format(m1))
    ax.legend()

    print("Order: {}".format(m0))
    print("Order: {}".format(m1))
   
    #plt.show()
    fig.savefig('t10_timesteps_by_half.png')

if __name__=='__main__':
    main()
