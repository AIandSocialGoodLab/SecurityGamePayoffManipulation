# coding: utf-8

# In[56]:

# ORIGAMI Alg, python version from Ryan's matlab code
# ORIGAMI Algorithm, as provided in [Kiekintveld et. al. 09]
# INPUT: `Params` is a 4-by-n matrix, the rows
#        `r': number of defend resources
# represent R^d, P^d, R^a, P^a. 
# The sorting w.r.t. R^a is also done inside the function

# OUTPUT: `coverage` is the coverage distribution, the index is in the
# original (unsorted) order
import numpy as np

DEBUG = False

def db(*args):
    if DEBUG: print(*args)

def origami_bs(Params, r=1):
    n = Params.shape[1]
    if (n == 1):
        return [1]
    ra = Params[2, :]
    # var below stores the mapping from the sorted list to the original list
    index_in_original_list = [b[0] for b in 
                              sorted(enumerate(ra), reverse=True, key=lambda i:i[1])]
    index_in_original_list = np.asarray(index_in_original_list)
    rds, pds, ras, pas = Params[:,index_in_original_list]
    db(ras)
    db(pas)
    start, end = 0, n
    lastTarget = n-1
    while(end - start > 1):
        # invariant: [start, end) contains the last target in the attack set
        mid = (start + end) // 2
        M, c = getUtilAndCov(ras, pas, mid, r)
        db('mid=%d, M=%f' % (mid, M))
        for i in range(n):
            db('c[%d]=%f' % (i, c[i]))
        for j in range(mid):
            if(c[j] < 0): 
                # attack set too large
                end = mid
                break
        if (end == mid): continue
        if(M < ras[mid]):
            # attack set too small
            start = mid
        else:
            # attack set is the right size, i.e. [0, mid) attack set (could be some
            # target outside after this having their penalty == attEU
            lastTarget = mid - 1 # last target in the attack set
            db('last target = %d' % lastTarget)
            for j in range(mid, n):
                if ras[j] == M:
                    lastTarget = j
                else:
                    db('last target = %d' % lastTarget)
                    break
            break
        db("Solution in [%d, %d)" % (start, end))

    M, c = getUtilAndCov(ras, pas, lastTarget+1, r)
    db('max attEU=%f' % M)
    for i in range(n):
        db('c[%d] = %f' % (i, c[i]))
    covBound = -np.inf
    for j in range(lastTarget+1):
        if c[j] >= 1: 
            covBound = np.max( (covBound, pas[j]) )
    if covBound > -np.inf:
        for j in range(lastTarget+1):
            c[j] = (ras[j] - covBound) / (ras[j] - pas[j]) 
    defEU = rds*c + pds*(1-c)
    #print("max: %f, argmax: %d" % (np.max(defEU), index_in_original_list[np.argmax(defEU)]))
    sol = np.zeros(n)
    for j in range(n):
        sol[index_in_original_list[j]] = c[j]
    return np.max(defEU), sol

def getUtilAndCov(R, P, k, r):
    # k is the first target outside attack set
    n = len(R)
    D = 1/(R - P)
    Dsum = np.sum(D[:k])
    RDprod = R * D
    RDsum = np.sum(RDprod[:k])
    M = (RDsum - r) / Dsum
    c = np.zeros(n)
    c[:k] = (R[:k]*Dsum - RDsum + r) / Dsum * D[:k]
    return M, c
    


# In[81]:

'''
# Test purpose
Params = np.abs(np.random.random((4,10)))
Params[[1,3], :] = -Params[[1,3],:]
defEU, index = origami_bs(Params)
print(defEU, index)
Params[[1,3], :] = -Params[[1,3],:]
ra = Params[2, :]
# var below stores the mapping from the sorted list to the original list
index_in_original_list = [b[0] for b in 
                          sorted(enumerate(ra), reverse=True, key=lambda i:i[1])]
rds, pds, ras, pas = Params[:,index_in_original_list]
'''

