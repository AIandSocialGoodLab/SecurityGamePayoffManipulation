% MILP for L^0 payoff manipulation [Shi et al., 2017], appendix
% Input:
% n: number of targets
% Params: 4xn positive payoff matrix (negation done inside the function)
% B: budget
% m: defensive resource
% Output: 
% finalRemove: binary vector, the targets being removed in the final solution
% finalZero: binary vector, the target whose P^a is set to 0 in the final solution
% fval: defender EU of the final solution
% x: entire optimal solution vector of the MILP
% x = [c, a, b, w, U_theta, U_Psi, d, k]

function [finalRemove, finalSetZero, fval,x] = oneMILP(n, Params, B, m)
    nVar = 6*n + 2;
    nIneq = 9*n + 2;
    nEq = n+1;
    Z = 1000000;
    Rd = Params(1,:);
    Pd = -Params(2,:);
    Ra = Params(3,:);
    Pa = -Params(4,:);

    A = zeros(nIneq, nVar);
    b = zeros(nIneq, 1);
    Aeq = zeros(nEq, nVar);
    beq = zeros(nEq, 1);
    intcon = (n+1):(4*n);
    lb = zeros(nVar, 1);
    ub = ones(nVar, 1);
    f = zeros(nVar, 1);
    for i=1:n
        A(i,n+i) = Z;
        A(i,6*n+1) = 1;
        A(i,4*n+i) = -1;
        b(i) = Z;
    end

    for i=1:n
        A(n+i,6*n+2) = -1;
        A(n+i,5*n+i) = 1;
        A(n+i,3*n+i) = -Z;
        b(n+i) = 0;
    end

    for i=1:n
        A(2*n+i,6*n+2) = 1;
        A(2*n+i,n+i) = Z;
        A(2*n+i,5*n+i) = -1;
        b(2*n+i) = Z;
    end
    % eq 24
    for i=1:n
        A(3*n+i,i) = Pa(i) - Ra(i);
        A(3*n+i,2*n+i) = -Z;
        A(3*n+i,5*n+i) = -1;
        b(3*n+i) = -Ra(i);
    end

    for i=1:n
        A(4*n+i,i) = Ra(i) - Pa(i);
        A(4*n+i,2*n+i) = -Z;
        A(4*n+i,5*n+i) = 1;
        b(4*n+i) = Ra(i);
    end

    for i=1:n
        A(5*n+i,i) = -Ra(i);
        A(5*n+i,2*n+i) = -Pa(i);
        A(5*n+i,5*n+i) = -1;
        b(5*n+i) = -Ra(i) - Pa(i);
    end

    for i=1:n
        A(6*n+i,i) = Ra(i);
        A(6*n+i,2*n+i) = -Pa(i);
        A(6*n+i,5*n+i) = 1;
        b(6*n+i) = Ra(i) - Pa(i);
    end

    for i=1:n
        A(7*n+i,n+i) = -1;
        A(7*n+i,2*n+i) = 1;
        b(7*n+i) = 0;
    end

    for i=1:n
        A(8*n+i,n+i) = 1;
        A(8*n+i,3*n+i) = 1;
        b(8*n+i) = 1;
    end    

    A(9*n+1,1:n) = 1;
    b(9*n+1) = m;

    A(9*n+2,(2*n+1):(4*n)) = 1;
    b(9*n+2) = B;
    
    for i=1:n
        Aeq(i,i) = Pd(i) - Rd(i);
        Aeq(i,4*n+i) = 1;
        beq(i) = Pd(i);
    end

    Aeq(n+1,(n+1):(2*n)) = 1;
    beq(n+1) = 1;

    lb((4*n+1):nVar) = -inf;
    ub((4*n+1):nVar) = inf;
    f(6*n+1) = -1;
    

    options = cplexoptimset('Display', 'off','MaxTime', 300);
  
    ctype = [repmat('C',1,n), repmat('B',1,3*n),repmat('C',1,nVar-4*n)];
    [x,fval,exitflag,output] = cplexmilp(f,A,b,Aeq,beq,[],[],[],lb,ub,ctype,[],options);
    fval = -fval;
    finalRemove = x((3*n+1):(4*n));
    finalRemove = finalRemove';
    finalSetZero = x((2*n+1):(3*n));
    finalSetZero = finalSetZero';
end