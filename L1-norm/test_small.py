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
    # expr1: compare b&b, singleMILP, multiMILP efficiency
    sList = range(10, 60, 10)
    smallInstance = []
    # below, row0 is b&b, row1 is singleMILP, row2 is multiMILP, row3 is IPOPT
    sopt = np.zeros((4,len(sList)))
    stime = np.zeros((4,len(sList)))
    for i in range(len(sList)):
        n = sList[i]
        Params = initData(n, 1, 1+n*10)
        P = l1manip(Params)
        P.bandb()
        P.buildSingleMILP()
        P.buildMultiMILP()
        P.buildIPOPT()
        smallInstance.append(P)
        sopt[0,i], stime[0,i] = P.opt_bandb, P.time_bandb
        sopt[1,i], stime[1,i] = P.opt_singleMILP, P.time_singleMILP
        sopt[2,i], stime[2,i] = P.opt_multiMILP, P.time_multiMILP
        sopt[3,i], stime[3,i] = P.opt_ipopt, P.time_ipopt
        with open(DEST_SMALL, "wb") as f:
            pickle.dump(smallInstance, f)
        with open('smallStat.dat', "wb") as f:
            pickle.dump([sopt, stime], f)
    # plot: Runtime, sol quality
    pl.plot(sList, stime[0], 's', label="B&B", markersize = 5)
    pl.plot(sList, stime[1], '^', label="singleMILP", markersize = 5)
    pl.plot(sList, stime[2], 'd', label="multiMILP", markersize = 5)
    pl.plot(sList, stime[3], 'o', label="ipopt", markersize = 5)
    legend = pl.legend()
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    pl.savefig("time_small.pdf")
    pl.clf() 

    pl.plot(sList, sopt[0], 's', label="B&B", markersize = 5)
    pl.plot(sList, sopt[1], '^', label="singleMILP", markersize = 5)
    pl.plot(sList, sopt[2], 'd', label="multiMILP", markersize = 5)
    pl.plot(sList, sopt[3], 'o', label="ipopt", markersize = 5)
    legend = pl.legend()
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    pl.savefig("opt_small.pdf")
    pl.clf() 