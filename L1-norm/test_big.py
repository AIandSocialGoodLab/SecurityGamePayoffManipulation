from bandb import *
import pickle
import glob
import pylab as pl

DEST_SMALL = 'small.dat'
DEST_BIG = 'big.dat'
def getIndToPlot(opt):
    # opt is (n,) ndarray, be consistent
    return np.where(opt > -np.inf)[0]

if __name__ == '__main__':
    # big instance test
    bList = range(100, 1100, 100) 
    bigInstance = []
    bopt = np.zeros((3, len(bList)))
    btime = np.zeros((3,len(bList)))
    for i in range(len(bList)):
        n = bList[i]
        Params = initData(n, 1, 1+n*10)
        P = l1manip(Params)
        P.bandb()
        P.buildSingleMILP()
        P.buildIPOPT()
        bigInstance.append(P)
        bopt[0,i], btime[0,i] = P.opt_bandb, P.time_bandb
        bopt[1,i], btime[1,i] = P.opt_singleMILP, P.time_singleMILP
        bopt[2,i], btime[2,i] = P.opt_ipopt, P.time_ipopt
        with open(DEST_BIG, "wb") as f:
            pickle.dump(bigInstance, f)
        with open('bigStat.dat', "wb") as f:
            pickle.dump([bopt, btime], f)

    pl.plot(bList, btime[0], 's', label="B&B", markersize = 5)
    pl.plot(bList, btime[1], '^', label="singleMILP", markersize = 5)
    pl.plot(bList, btime[2], 'd', label="ipopt", markersize = 5)
    legend = pl.legend()
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    pl.savefig("time_big.pdf")
    pl.clf() 
    
    with open('bigStat.dat', "rb") as f:
        bopt, btime = pickle.load(f)
    with open(DEST_BIG, "rb") as f:
        bigInstance = pickle.load(f)
    bList = np.arange(100, 1100, 100) 
    l0 = getIndToPlot(bopt[0]) 
    print(l0)
    pl.plot(bList[l0], bopt[0][l0], 's', label="B&B", markersize = 5)

    l1 = getIndToPlot(bopt[1]) 
    pl.plot(bList[l1], bopt[1][l1], '^', label="singleMILP", markersize = 5)

    l2 = getIndToPlot(bopt[2])
    pl.plot(bList[l2], bopt[2][l2], 'd', label="ipopt", markersize = 5)
    legend = pl.legend()
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    pl.savefig("opt_big.pdf")
    pl.clf() 