import gurobipy as grb
import numpy as np
import copy
import origami
import ipoptsolve as ipopt
from pyomo.opt import TerminationCondition
from pyomo.environ import *
import l1d2 as origamiMILP
import time


DEBUG = False
GENERAL = True
TIMELIMIT = 900 # time limit for MILP
def db(*args):
    if DEBUG: print(*args)

class initData:
    def __init__(self, n, lo, hi):
        self.payoff = np.random.randint(lo, hi, (4,n))
        self.payoff[1,:] = -self.payoff[1,:]
        self.payoff[3,:] = -self.payoff[3,:]
        if GENERAL:
            self.B = np.sum(self.payoff[2,:]-self.payoff[3,:]) # huge budget
            db('Budget = %d' % self.B)
            self.mu, self.theta = np.random.randint(1, self.B, (2,n))
            self.r = np.random.randint(1, 10)
        else:
            self.B = np.min(-self.payoff[3,:])
            self.mu, self.theta = np.ones((2,n))
            self.r = 1

class l1manip:
    def __init__(self, Params):
        # init Params:
        self.Params = Params
        self.payoff = Params.payoff
        self.rd = Params.payoff[0,:]
        self.pd = Params.payoff[1,:]
        self.ra = Params.payoff[2,:]
        self.pa = Params.payoff[3,:]
        self.mu = Params.mu # weight on manip of ra
        self.theta = Params.theta # weight on manip of pa
        self.B = Params.B # budget
        self.r = Params.r # total resource
        self.n = len(self.rd)
        self.N = set() # set of non-attack targets from pruning alg
        self.tol = 1e-4*np.min(self.rd) # multiplicative ratio 1e-4 in MILP-based approx.
        self.rho = self.tol*np.min(self.ra/(self.rd-self.pd)) / 2 # approx. value to set in MILP

        # init sols (store the best sol found so far)
        self.bestVal = -np.inf
        self.sol = np.ones((3,self.n)) # 1st, 2nd, 3rd is c, eps, delta resp.

        # test stats
        self.time_multiMILP, self.time_singleMILP, self.time_bandb = 0, 0, 0
        self.time_ipopt = 0
        self.opt_multiMILP, self.opt_singleMILP, self.opt_bandb = -np.inf, -np.inf, -np.inf
        self.opt_ipopt = -np.inf

    def getGlobalLB(self):
        # compute global lower bound by taking maximum over 
        # greedy manipulations of all subproblems
        print('\nComputing global LB...')
        lblist = -np.inf * np.ones(self.n)
        lbsol = []
        for i in range(self.n):
            lb, sol = self.getAPLB(i)
            lblist[i] = lb
            lbsol.append(sol)
            if lb > self.bestVal:
                print('AP%d increases globalLB from %f to %f' % (i, self.bestVal, lb))
                self.bestVal = lb
                self.sol = sol
        print('\nGlobal LB obtained, LB = %f' % self.bestVal)
        db('lbList: %s' % lblist)
        for i in range(self.n):
            db('c[%d]=%f, eps[%d]=%f, delta[%d]=%f' %
                  (i, self.sol[0,i], i, self.sol[1,i], i, self.sol[2,i]))
        sortedIndex = self.sortSubproblem(lblist)
        return sortedIndex, lbsol

    def sortSubproblem(self,A):
        # get a mapping from sorted index [1,..,n] to original index
        # i.e. result[i] = ith element in the original list
        # reverse = True for descending order
        return [b[0] for b in sorted(enumerate(A), reverse=True, key=lambda i:i[1])]

    def getAPLB(self, i):
        # compute LB for AP_i using greedy manip
        # GM1: R^i += B
        GM1, GM2 = self.getGM(i)
        val1, c1 = origami.origami_bs(self.payoff + GM1, self.r)
        val2, c2 = origami.origami_bs(self.payoff + GM2, self.r)
        if val1 > val2:
            #db('AP%d GM1 better sol' % i)
            return val1, np.vstack((c1, GM1[2:,:]))
        else:
            #db('AP%d GM2 better sol' % i)
            return val2, np.vstack((c2, GM2[2:,:]))

    def getGM(self, i):
        # greedy mod 
        # 1. increase R_i AMAP
        gm1 = np.zeros((4, self.n))
        gm1[2,i] = self.B // self.mu[i]
        # 2. only increase P_i until hits zero, then increase R_i
        gm2 = np.zeros((4, self.n))
        gm2[3,i] = np.min( (-self.pa[i], self.B//self.theta[i]) )
        remainB = np.max( (0, self.B - self.theta[i]*gm2[3,i]) )
        gm2[2,i] = remainB // self.mu[i]
        return gm1, gm2

    def getAPUB(self, i):
        # compute upper bound for subproblem i by 
        # reusing budget to manipulate all targets 
        # not in the non-attack set N, which contains 
        # the set of targets pruned before
        payoffReuse = copy.deepcopy(self.payoff)
        for j in range(self.n):
            if j not in self.N and j != i:
                payoffReuse[2,j] = np.max((0, payoffReuse[2,j]-self.B//self.mu[j]))
                payoffReuse[3,j] -= self.B//self.theta[j]
        targetManip1, targetManip2 = self.getGM(i)
        opt1, _ = origami.origami_bs(payoffReuse + targetManip1, self.r)
        opt2, _ = origami.origami_bs(payoffReuse + targetManip2, self.r)
        return np.max((opt1, opt2))

    def buildMILP(self, i):
        n = self.n
        B = self.B
        R = copy.deepcopy(self.ra)
        P = copy.deepcopy(self.pa)
        sumindR = np.ceil(np.log2((B + R)/self.rho))+1 # summation range for each target of R
        sumindP = np.ceil(np.log2((B - P)/self.rho))+1 # similarly as above
        sumindR, sumindP = sumindR.astype(int), sumindP.astype(int)

        # build gurobi model
        m = grb.Model("AP%d"%i)
        #m.setParam('OutputFlag', False)
        # create eps, delta as the perturbation variable
    #    eps_abs, delta_abs = m.addVars(n), m.addVars(n)
        eps, delta = m.addVars(n, lb = -grb.GRB.INFINITY, ub=0), m.addVars(n, lb = -grb.GRB.INFINITY, ub=0)
        eps[i].lb, eps[i].ub = 0, grb.GRB.INFINITY
        delta[i].lb, delta[i].ub = 0, grb.GRB.INFINITY
        c = m.addVars(n) # mixed strategy probability
        v = m.addVars(n) # attacker utility
        # create y,z as a list of tupledict (grb obj)
        y, z = [], []
        alpha, beta = [], []
        m.addConstr(P[i] + delta[i] <= 0)
        for j in range(n):
            y.append(m.addVars(range(sumindR[j]), vtype = grb.GRB.BINARY,
                    name = "r-perturb%d"%j))
            z.append(m.addVars(range(sumindP[j]), vtype = grb.GRB.BINARY,
                    name = "p-perturb%d"%j))
            alpha.append(m.addVars(range(sumindR[j]), name = "alpha%d"%j))
            beta.append(m.addVars(range(sumindP[j]), name = "beta%d"%j))
        # add constrs of all types except (6)&(12)&(14), refer to writeup
        # constr (2)&(3):
            # generate binary sequance [2**i] for binary representation
            binary_R = [2**k for k in range(sumindR[j])]
            binary_P = [2**k for k in range(sumindP[j])]
            binRdict = dict(enumerate(binary_R))
            binPdict = dict(enumerate(binary_P))
            # the above bin dict is reusable in constr (11)
            m.addConstr(eps[j] == self.rho * y[j].prod(binRdict) - R[j])
            m.addConstr(delta[j] == -self.rho * z[j].prod(binPdict) - P[j])
        # constr (7)&(8):
            if j != i:
                m.addConstr(R[j] + eps[j] >= 0)
            #m.addConstr(P[j] + delta[j] <= 0)
        # constr (9)&(10):
            m.addConstrs(alpha[j][k] <= y[j][k] for k in range(sumindR[j]))
            m.addConstrs(c[j] - 1 + y[j][k] <= alpha[j][k] for k in range(sumindR[j]))
            m.addConstrs(alpha[j][k] <= c[j] for k in range(sumindR[j]))
            m.addConstrs(beta[j][k] <= z[j][k] for k in range(sumindP[j]))
            m.addConstrs(c[j] - 1 + z[j][k] <= beta[j][k] for k in range(sumindP[j]))
            m.addConstrs(beta[j][k] <= c[j] for k in range(sumindP[j]))
        # constr (11):
            m.addConstr(v[j] == R[j] + eps[j] - self.rho * (alpha[j].prod(binRdict)
                       + beta[j].prod(binPdict)))
        # constr (6):
        mu, theta = copy.deepcopy(-self.mu), copy.deepcopy(-self.theta)
        mu[i] = -mu[i]
        theta[i] = -theta[i]
        coeff_mu, coeff_theta = dict(enumerate(mu)), dict(enumerate(theta))
        m.addConstr(eps.prod(coeff_mu) + delta.prod(coeff_theta) <= B)
        # constr (12):
        m.addConstrs(v[i] >= v[j] for j in range(n))
        # constr (14)
        m.addConstr(c.sum() <= self.r)
        # set objective
        m.setObjective(self.rd[i]*c[i]+self.pd[i]*(1-c[i]), grb.GRB.MAXIMIZE)
        m._sol = [] 
        m._sol.append(c)
        m._sol.append(eps)
        m._sol.append(delta)
        m._v = v
        return m

    def buildSingleMILP(self):
        print('\nStart building single MILP...')
        m = origamiMILP.singleMILP(self.Params, self.rho)
        m.setParam('TimeLimit', TIMELIMIT)
        m.setParam('MIPGap', 5e-3)
        m.optimize()
        self.time_singleMILP = m.Runtime
        if(m.status == grb.GRB.OPTIMAL):
            self.opt_singleMILP = m.ObjVal 
            for i in range(self.n):
                db('c[%d]=%f, eps[%d]=%f, delta[%d]=%f' % 
                   (i, m._c[i].X, i, m._eps[i].X, i, m._delta[i].X))

    def buildMultiMILP(self):
        for i in range(self.n):
            db('Solving AP%d' % i)
            m = self.buildMILP(i)
            m.setParam('TimeLimit', TIMELIMIT)
            m.setParam('MIPGap', 5e-3)
            m.optimize()
            self.time_multiMILP += m.Runtime
            if (m.status == grb.GRB.OPTIMAL):
                self.opt_multiMILP = np.max((self.opt_multiMILP, m.ObjVal))

    def updateSol(self, model):
        # input: gurobi model
        self.bestVal = model.ObjVal
        for i in range(3):
            for j in range(self.n):
                self.sol[i,j] = model._sol[i][j].X

    def initModel(self, m, sol):
        m.setParam('TimeLimit', TIMELIMIT)
        m.setParam('MIPGap', 5e-3)
        # sol[0,:] = c, sol[1,:] = eps, sol[2,:] =delta
        for i in range(sol.shape[0]):
            for j in range(sol.shape[1]):
                m._sol[i][j].start = sol[i,j]
        return m

    def buildIPOPT(self):
        #optList = -np.inf * np.ones(self.n)
        for i in range(self.n):
            m = ipopt.l1continuous_subproblem(self.Params, i)
            if self.opt_ipopt > -np.inf:
                m.lbConstr = Constraint(
                    expr = self.rd[i]-(self.rd[i]-self.pd[i])*m.c[i]>=self.opt_ipopt)
            solver = pyomo.opt.SolverFactory('ipopt')
            solver.options['linear_solver'] = 'ma86'
            solver.options['tol'] = 1e-10
            result = solver.solve(m, tee = True)
            self.time_ipopt += result.solver.time
            if result.solver.termination_condition == TerminationCondition.optimal:
                self.opt_ipopt = np.max((self.opt_ipopt, m.obj()))
        print('best by IPOPT: %f' % self.opt_ipopt)


    def bandb(self):
        def stopByGlobalLB(model, where):
            if where == grb.GRB.Callback.MIP:
                objBnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
                if abs((objBnd - self.bestVal)/self.bestVal) <= 1e-2:
                    print('Stop early: close to global LB')
                    model.terminate()
        start = time.time()
        spList, solList = self.getGlobalLB()
        print('\nStarting B&B... global LB = %f' % self.bestVal)
        for i in spList:
            # invariant: SP[0, i] either pruned, or approxed by MILP, 
            # gLB is the best sol found so far
            ub = self.getAPUB(i)
            print('SP%d ub = %f' % (i,ub))
            if ub <= self.bestVal:
                print('SP%d pruned: UB <= globalLB\n' % i)
                self.N.add(i)
                continue
            else:
                db(self.ra)
                db(self.pa)
                model = self.buildMILP(i)
                model = self.initModel(model, solList[i])
                # add a cut:
                model.addConstr(self.rd[i] * model._sol[0][i] + self.pd[i] * (1-model._sol[0][i]) >= self.bestVal)
                model.optimize(stopByGlobalLB)
                if model.status == grb.GRB.OPTIMAL and self.bestVal < model.ObjVal:
                    print('SP%d feasible, updating LB from %f to %f\n'
                       % (i, self.bestVal, model.ObjVal))
                    for j in range(self.n):
                        db('attEU[%d] = %f' % (j,model._v[j].X))
                    self.updateSol(model)
                else: 
                    print('SP%d stopped in MILP-approx\n' % i)
        end = time.time()
        self.time_bandb = end-start
        self.opt_bandb = self.bestVal
        print('B&B bestVal = %f' % self.bestVal)

if __name__ == '__main__':
    # generate payoff matrix
    n = 100
    lo = 100
    Params = initData(n, lo, lo+n*10)
    P = l1manip(Params)
    P.bandb()
    '''
    P.buildSingleMILP()
    P.bandb()
    print('OPT = %f' % P.bestVal)
    print('Pruning list = %s' % P.N)
    for i in range(n):
        print('c[%d]=%f, eps[%d]=%f, delta[%d]=%f' %
              (i, P.sol[0,i], i, P.sol[1,i], i, P.sol[2,i]))
    P.getGlobalLB()
    print('globalLB = %f\n' % P.bestVal)
    for i in range(n):
        print('AP%d UB = %f' %(i, P.getAPUB(i)))
        print('AP%d LB = %f\n' %(i, P.getAPLB(i)[0]))
    m = P.buildMILP(0)
    m.optimize()
    P.getGlobalLB()
    print(P.bestVal)
    for i in range(P.n):
        print('c[%d]=%f, eps[%d]=%f, delta[%d]=%f' % 
              (i, P.sol[0,i], i, P.sol[1,i], i, P.sol[2,i]))
    '''
