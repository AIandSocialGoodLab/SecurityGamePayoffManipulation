
Requires IBM ILOG CPLEX Optimization Studio 12.8+



This repo contains the code for reproducing experiments for L0-norm case.

Code corresponding to the algorithms introduced in the paper, with dependences.

- O(n^3):  polyTime.m  <-- randomSelect.m
- MILP: oneMILP.m
- L0-Greedy1: greedy.m  <-- origami.m
- L0-Greedy2: greedy2.m  <-- origami.m



There are two main experiments, in test.m.

We compare O(n^3), MILP, and two greedy algorithms in terms of runtime and solution (multiplicative) gap
w.r.t. O(n^3) and MILP on instance sizes = 50, 100, ..., 250, with 
1. m=1 defensive resource, and
2. m=n/10 defensive resources

Any feedback or bug reports are welcome.

Prerequisite:
IBM ILOG CPLEX Optimization Studio 12.8+ [[https://www.ibm.com/products/ilog-cplex-optimization-studio]](https://www.ibm.com/products/ilog-cplex-optimization-studio)
for solving MILPs.
