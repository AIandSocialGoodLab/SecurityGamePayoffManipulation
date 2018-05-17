% O(n^3) algorithm for L^0 payoff manipulation [Shi et al., 2017]
% Input:
% n: number of targets
% Params: 4xn positive payoff matrix (negation done inside the function)
% B: budget
% m: defensive resource
% Output: 
% candidates: nxn matrix of defender EU for each subproblem
% fval: the maximum entry in candidates, i.e. overall optimal value

function [candidates, fval] = polyTime(n, Params, B, m)
Params(2,:) = -Params(2,:);
Params(4,:) = -Params(4,:);
Params = sortrows(Params',3,'descend')';
Rd = Params(1,:);
Pd = Params(2,:);
Ra = Params(3,:);
Pa = Params(4,:);
RaExt = [Ra,-inf];
D = 1./(Ra-Pa);
candidates = ones(n)*(-inf);
for k=2:n  %attack set
    for i=1:k   %attack target
        Dcp = D;
        numRmv = min(B, k-1);
        V = (Ra(i) - Ra) .* D + m/(k - numRmv - 1);
        V = [V(1:i-1),V(i+1:k)];
        W = [D(1:i-1),D(i+1:k)] + D(i)/(k - numRmv - 1);
        Astar = randomSelect(V, W, numRmv);
        f = V - Astar * W;
        [dummy,remove] = mink(f, numRmv);
        remove(remove >= i) = remove(remove >= i) + 1;
        Dcp(remove) = 0;
        sumD = sum(Dcp(1:k));
        sumDR = sum(Dcp(1:k) .* Ra(1:k));
        c = Dcp(1:k) .* (Ra(1:k)*sumD - sumDR + m)/sumD;
        Value = (sumDR - m)/sumD;
        if all(c >= 0) && all(c <= 1) && Value >= RaExt(k+1)
            candidates(k,i) = (Rd(i) - Pd(i))* (Astar*Dcp(i)) + Pd(i);
        end
        
        numRmv = min(B-1, k-1);
        Dcp = D;
        V = (Ra(i) - Ra) .* D + m/(k - numRmv - 1);
        V = [V(1:i-1),V(i+1:k)];
        W = [D(1:i-1),D(i+1:k)] + (1/Ra(i))/(k - numRmv - 1);
        Astar = randomSelect(V, W, numRmv);
        f = V - Astar * W;
        [dummy,remove] = mink(f, numRmv);
        remove(remove >= i) = remove(remove >= i) + 1;
        Dcp(remove) = 0;
        Dcp(i) = 1/Ra(i);
        sumD = sum(Dcp(1:k));
        sumDR = sum(Dcp(1:k) .* Ra(1:k));
        c = Dcp(1:k) .* (Ra(1:k)*sumD - sumDR + m)/sumD;
        Value = (sumDR - m)/sumD;
        if all(c >= 0) && all(c <= 1) && Value >= RaExt(k+1)
            candidates(k,i) = max(candidates(k,i), (Rd(i) - Pd(i))* (Astar*Dcp(i)) + Pd(i));
        end
    end
end

k = n; % attack set, largest, for coverage with certainty
numRmvInit = min(B, k-1);
[dummy,possiblel] = maxk(Pa, numRmvInit);
for li=1:numRmvInit
    l = possiblel(li);
    for i = 1:k % attack target
        numRmv = numRmvInit - li + 1;
        c = (Pa(l) - Ra) ./ (Pa - Ra);
        c(possiblel(1:li-1)) = 0;
        [dummy, remove] = maxk(c, numRmv+1);
        if any(remove == i) && i ~= l
            [dummy, remove] = maxk(c, numRmv+2);
        end
        finc = c;
        finc(remove) = 0;
        finc(l) = c(l);
        finc(i) = c(i);
        if sum(finc) <= m
            candidates(k,i) = max(candidates(k,i), finc(i) * Rd(i) + (1-finc(i)) * Pd(i));
        end
    end
end

for i=1:k
    l = i;
    Pacp = Pa;
    Pacp(i) = 0;
    numRmv = min(B-1, k-1);
    c = (Pacp(l) - Ra) ./ (Pacp - Ra);
    [dummy, remove] = maxk(c, numRmv+1);
    finc = c;
    finc(remove) = 0;
    finc(l) = c(l);
    if sum(finc) <= m
        candidates(k,i) = max(candidates(k,i), finc(i) * Rd(i) + (1-finc(i)) * Pd(i));
    end
end

finalSetZero = [];
fval = max(candidates(:));
end
        
        