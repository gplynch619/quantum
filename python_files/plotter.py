import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def read_data(filename):
    d=defaultdict(list)
    csv_file=open(filename)
    reader=csv.DictReader(csv_file)
    for l in reader:
        for k,v in l.items():
            d[k].append(v)
    return d

def main():
    file1="analysis_data/4000duration.csv"
    file2="analysis_data/4000zoomin.csv"
    file3="analysis_data/4000zoomin2.csv"
    file4="analysis_data/4000zoomin3.csv"
    #Read Data#
    d = read_data(file1)
    dt1=np.array([float(i) for i in d["dt"]])
    E1=np.array([float(i) for i in d["E"]])
    T1=np.linspace(start=512, stop=51200, num=100, endpoint=True)
    
    d = read_data(file2)
    dt2=np.array([float(i) for i in d["dt"]])
    E2=np.array([float(i) for i in d["E"]])
    T2=np.array([float(i) for i in d["T"]])
    
    d = read_data(file3)
    dt3=np.array([float(i) for i in d["dt"]])
    E3=np.array([float(i) for i in d["E"]])
    T3=np.array([float(i) for i in d["T"]])
    
    d = read_data(file4)
    dt4=np.array([float(i) for i in d["dt"]])
    E4=np.array([float(i) for i in d["E"]])
    T4=np.array([float(i) for i in d["T"]])
    ###########
    #Plot#
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(1,1)

    run1, =ax.plot(dt1, E1, marker='.', linestyle='None')#, label="Sample=40")
    run2, =ax.plot(dt2, E2, marker='.', linestyle='None')#, label="Sample=40")
    run3, =ax.plot(dt3, E3, marker='.', linestyle='None')#, label="Sample=40")
    run4, =ax.plot(dt4, E4, marker='x', linestyle='None')#, label="Sample=40")
    ax.set_title("Estimated energy vs. time step, $N_t \in [4000, 8000], T=4000$", fontsize=14)
    ax.set_xlabel("$\Delta$t", fontsize=14)
    ax.set_ylabel("E", fontsize=14)
    #ax.set_ylim(bottom=-0.54)

    ax.axhline(y=-0.5, color='C7', linestyle='--')

    #ax.legend(handles=[run1], fontsize=14)
    ###########
    #Calculate convergence#
    print(np.min(T1))
    print(dt1[2])
    delt=np.flip(dt1)
    E=np.flip(E1)
    estimates=[]
    for step in np.arange((delt.size-2)):
        num=np.abs(E[step] - E[step+1])
        den=np.abs(E[step+1] - E[step+2])
        est=num/den
        estimates.append(est)
    
    estimates=np.array(estimates)
    with open("ratio_test.out", 'w') as filename:
        filename.write("dt\tE\tratio\tlog2(ratio)\n")
        for i, est in enumerate(estimates):
            order=np.log2(est)
            filename.write("{0}\t{1}\t{2}\t{3}\n".format(delt[i], E[i], est, order))

    plt.show()
if __name__=="__main__":
    main()
