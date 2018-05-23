import gurobipy as grb # optimizer
import numpy as np

Z = 1e10 # large number in MILP formulation

def singleMILP(Params, rho = 1):
    # This function takes in the following input:
    # R -- 1darray of rewards
    # P -- 1darray of penalties
    # B -- total budget, scalar
    # mu -- weight of reward perturbation
    # theta -- weight of penalty perturbation
    # rho -- atomic change threshold
    # then builds a MILP and solves it
    
    # check inputs correctness

    payoff = Params.payoff
    R = payoff[2,:]
    P = payoff[3,:]
    B = Params.B
    mu = Params.mu
    theta = Params.theta
    try:
        assert(R.shape == P.shape)
        assert(R.min() >= 0 and P.max() <= 0 and B >= 0)
        assert(len(mu) == len(theta) and len(mu) == len(R))
    except AssertionError as error:
        print("Invalid Input!")
    
    # define const
    n = len(R) # number of targets
    sumindR = np.ceil(np.log2((B + R)/rho))+1 # summation range for each target of R
    sumindP = np.ceil(np.log2((B - P)/rho))+1 # similarly as above
    sumindR, sumindP = sumindR.astype(int), sumindP.astype(int)

    # build gurobi model
    m = grb.Model("l1-discrete")
    #m.setParam('OutputFlag', False)
    # create eps, delta as the perturbation variable
    eps_abs, delta_abs = m.addVars(n), m.addVars(n)
    eps, delta = m.addVars(n, lb = -grb.GRB.INFINITY), m.addVars(n, lb = -grb.GRB.INFINITY)
    #eps, delta = m.addVars(n, lb = -grb.GRB.INFINITY, ub=0), m.addVars(n, lb = -grb.GRB.INFINITY, ub=0)
    #eps[i].lb, eps[i].ub = 0, grb.GRB.INFINITY
    #delta[i].lb, delta[i].ub = 0, grb.GRB.INFINITY
    c = m.addVars(n) # mixed strategy probability
    v = m.addVars(n) # attacker utility
    # create y,z as a list of tupledict (grb obj)
    y, z = [], []
    alpha, beta = [], []
    gamma = m.addVars(n, vtype=grb.GRB.BINARY)
    U = m.addVar(lb = -grb.GRB.INFINITY)
    d = m.addVar(lb = -grb.GRB.INFINITY)
    #m.addConstr(P[i] + delta[i] <= 0)
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
        m.addConstr(eps[j] == rho*y[j].prod(binRdict) - R[j])
        m.addConstr(delta[j] == -rho*z[j].prod(binPdict) - P[j])
    # constr (4)&(5):
        m.addConstr(eps[j] <= eps_abs[j])
        m.addConstr(eps[j] >= -eps_abs[j])
        m.addConstr(delta[j] <= delta_abs[j])
        m.addConstr(delta[j] >= -delta_abs[j])
    # constr (7)&(8):
        #if j != i:
        #    m.addConstr(R[j] + eps[j] >= 0)
        #m.addConstr(P[j] + delta[j] <= 0)
        m.addConstr(R[j] + eps[j] >= 0)
        m.addConstr(P[j] + delta[j] <= 0)
    # constr (9)&(10):
        m.addConstrs(alpha[j][k] <= y[j][k] for k in range(sumindR[j]))
        m.addConstrs(c[j] - 1 + y[j][k] <= alpha[j][k] for k in range(sumindR[j]))
        m.addConstrs(alpha[j][k] <= c[j] for k in range(sumindR[j]))
        m.addConstrs(beta[j][k] <= z[j][k] for k in range(sumindP[j]))
        m.addConstrs(c[j] - 1 + z[j][k] <= beta[j][k] for k in range(sumindP[j]))
        m.addConstrs(beta[j][k] <= c[j] for k in range(sumindP[j]))
    # constr (11):
        m.addConstr(v[j] == R[j] + eps[j] - rho*(alpha[j].prod(binRdict)
                   + beta[j].prod(binPdict)))
        m.addConstr(v[j] <= U)
        m.addConstr(v[j] >= U - (1-gamma[j]) * Z)
        m.addConstr(d <= payoff[0,j]*c[j] + payoff[1,j]*(1-c[j]) + (1-gamma[j])*Z)

   
    # constr (6):
    #mu, theta = -mu, -theta
    #mu[i] = -mu[i]
    #theta[i] = -theta[i]
    coeff_mu, coeff_theta = dict(enumerate(mu)), dict(enumerate(theta))
    m.addConstr(eps_abs.prod(coeff_mu) + delta_abs.prod(coeff_theta) <= B)
    
    m.addConstr(gamma.sum() == 1)
    # constr (12):
    #m.addConstrs(v[i] >= v[j] for j in range(n))
    
    # constr (14)
    m.addConstr(c.sum() <= Params.r)
    
    # set objective
    m.setObjective(d, grb.GRB.MAXIMIZE)
    m._eps = eps 
    #m._epsabs = eps_abs
    m._delta = delta 
    #m._deltaabs = delta_abs
    m._c = c
    m._v = v
    m._sol = [] 
    m._sol.append(c)
    m._sol.append(eps)
    m._sol.append(delta)
    return m

'''
def l1discrete(R, P, B, mu, theta):
    # check inputs correctness
    try:
        assert(R.shape == P.shape)
        assert(R.min() >= 0 and P.max() <= 0 and B >= 0)
        assert(len(mu) == len(theta) and len(mu) == len(R))
    except AssertionError as error:
        print("Invalid Input!")
        
    optlist = np.zeros(len(R)) # stores optima for each subproblem
    tlist = np.zeros(len(R)) # document time for each subproblem solving
    for i in range(len(R)):
        m = l1discrete_subproblem(R, P, B, mu, theta, i)
        m.optimize()
        if m.Status == grb.GRB.OPTIMAL:
            optlist[i] = m.ObjVal
        tlist[i] = m.Runtime
    return (optlist, tlist)

# In[154]:

np.random.seed(1)
n = 100
scale = 1000
R = (scale*np.random.rand(n)).astype(int)
P = -(scale*np.random.rand(n)).astype(int)
print(R)
print(P)
mu = np.ones(n)
theta = np.ones(n)
timelist = []
result = []
for B in [scale//20, scale//10, scale//5, scale]:
    optlist, tlist = l1discrete(R,P,B,mu,theta)
    result.append(optlist)
    timelist.append(tlist)


# In[163]:

result[0]


# In[148]:

# plot stats
import glob
from pylab import *
params = {
   'axes.labelsize': 8,
   'text.fontsize': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [4.5, 4.5]
   }
rcParams.update(params)
x = np.arange(0, n)
for i in range(len(timelist)):
    plot(x, timelist[i], linewidth=2)
legend = legend(["Budget = %d"%(scale//20), "Budget = %d"%(scale//10), 
                 "Budget = %d"%(scale//5), "Budget = %d"%scale]);
frame = legend.get_frame()
frame.set_facecolor('0.9')
frame.set_edgecolor('0.9')
#show()
savefig('l1discrete.png')
'''
