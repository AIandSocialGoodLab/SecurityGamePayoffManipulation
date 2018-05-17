% L0-Greedy1 algorithm for L^0 payoff manipulation [Shi et al., 2018], appendix
% Input:
% Params: 4xn positive payoff matrix, n is the number of targets
% B: budget
% Output: 
% finalRemove: binary vector, the targets being removed in the final solution
% finalZero: target whose P^a is set to 0 in the final solution, 0 if none
% fval: defender EU of the final solution

function [finalRemove, finalZero, fval] = greedy(Params, B)
n = size(Params, 2);
Params = [Params; 1:n];
finalZero = 0;
finalRemove = zeros(1,n);
for i=1:B
    nLeft = size(Params, 2);
    defEUs = -inf*ones(2, nLeft);
    for t=1:nLeft
        if t ~= finalZero
            tParams = cat(2, Params(:,1:t-1), Params(:,t+1:end));
            tc = origami(tParams);
            defEUt = tParams(1,:) .* tc - tParams(2,:) .* (1-tc);
            defEUt(~logical(tc)) = -inf;
            defEUs(1,t) = max(defEUt);
        end
    end
    if finalZero == 0
        for t=1:nLeft
            tParams = Params;
            tParams(4, t) = 0;
            tc = origami(tParams);
            defEUt = tParams(1,:) .* tc - tParams(2,:) .* (1-tc);
            defEUt(~logical(tc)) = -inf;
            defEUs(2,t) = max(defEUt);
        end
    end
    [removeMax, removeIndex] = max(defEUs(1,:));
    [zeroMax, zeroIndex] = max(defEUs(2,:));
    if removeMax >= zeroMax
        finalRemove(Params(end,removeIndex)) = 1;
        Params = cat(2, Params(:,1:removeIndex-1), Params(:, removeIndex+1:end));
    else
        finalZero = Params(end,zeroIndex); 
        Params(4, zeroIndex) = 0;
    end
end
fval = max(removeMax, zeroMax);
