import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    load_file=np.load(sys.argv[1])
    ng=load_file['ng']
    rmse=load_file['rmse']
    dxs=load_file['dxs']

    ratios=[]
    for i in np.arange(0, rmse.shape[0]-1): 
        num=rmse[i]
        den=rmse[i+1]
        ratios.append(np.log2(num/den))
    ratios=np.array(ratios)
    print("Covergence order: ",ratios)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    fig, ax=plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Convergence order estimation, T=10")
    ax.set_xlabel("$\Delta$t", fontsize=14)
    ax.set_ylabel("$\delta$ (RMSE)", fontsize=14)
    
    ax.plot(dxs[0], rmse, 'x')
    ax.legend()

   
    plt.show()

    #fig.savefig('ts_100_to_5k_converge.png')
    #np.save('t10_ts_by_half_convergence.npz', ratios)
if __name__=='__main__':
    main()
