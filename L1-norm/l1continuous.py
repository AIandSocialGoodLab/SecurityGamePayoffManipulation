from __future__ import division
from pyomo.environ import *
from pyomo.opt import TerminationCondition
import numpy as np

def l1continuous_subproblem(R, P, B, mu, theta, i, init = None):
    # all input same as l1discrete_subproblem except
    # init is a dictionary having key: 'c', 'delta', 'eps'
    try:
        assert(R.shape == P.shape)
        assert(R.min() >= 0 and P.max() <= 0 and B >= 0)
        assert(len(mu) == len(theta) and len(mu) == len(R))
        assert(i>=0 and i<len(R))
    except AssertionError as error:
        print("Invalid Input!")
    n = len(R)
    model = ConcreteModel()
    model.c = Var(range(n), domain=NonNegativeReals, bounds=(0.0, 1.0)) # mixed strategy
    model.delta = Var(range(n), domain=NonNegativeReals) # change in penalty
    model.eps = Var(range(n), domain=NonNegativeReals) # change in reward
    # init primal var
    if init != None:
        print('init provided: %s\n' % init)
        for i in range(n):
            model.c[i].value = init['c'][i]
            model.eps[i].value = init['eps'][i]
            model.delta[i].value = init['delta'][i]
    else:
        print('No initialization provided...\n')
    model.obj = Objective(expr = model.c[i], sense = maximize) 
    model.highestUtility_i = ConstraintList()
    for j in range(n):
        if j != i:
            model.highestUtility_i.add(
                (R[i]+model.eps[i]) * (1-model.c[i]) + (P[i]+model.delta[i]) * model.c[i] >= 
                (R[j]-model.eps[j]) * (1-model.c[j]) + (P[j]-model.delta[j]) * model.c[j])
    model.rangeConstr = ConstraintList()
    model.rangeConstr.add(P[i] + model.delta[i] <= 0)
    for j in range(n):
        if j != i:
            model.rangeConstr.add(R[j] - model.eps[j] >= 0)
    model.BudgetConstr = Constraint(expr = sum(mu[j] * model.eps[j] + theta[j] * model.delta[j] for j in range(n)) <= B)
    model.simplexConstr = Constraint(expr = sum(model.c[j] for j in range(n)) == 1)
    model.linfConstr = ConstraintList()
    '''
    for i in range(n):
        model.linfConstr.add(model.eps[i] <= np.random.randint(low=1))
        model.linfConstr.add(model.delta[i] <= np.random.randint(low=1))
    '''
    solver = pyomo.opt.SolverFactory('ipopt')
    solver.options['linear_solver'] = 'ma86'
    solver.options['tol'] = 1e-10
    result = solver.solve(model, tee = False)
    return result, model
    
def l1continuous(R, P, B, mu, theta,init = None):
    # init is a list of python dictionary
    n = len(R)
    optlist = np.zeros(n) 
    tlist = np.zeros(n) 
    for i in range(n):
        if init != None:
            result, model = l1continuous_subproblem(R, P, B, mu, theta, i, init[i])
        else:
            print("subproblem %d has no init val." % i)
            result, model = l1continuous_subproblem(R, P, B, mu, theta, i)
        if result.solver.termination_condition == TerminationCondition.optimal:
            optlist[i] = model.obj()
        tlist[i] = result.solver.time
    return optlist, tlist
