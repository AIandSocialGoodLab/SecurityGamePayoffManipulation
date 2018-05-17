% L0-Greedy2 algorithm for L^0 payoff manipulation [Shi et al., 2017], appendix
% Input:
% Params: 4xn positive payoff matrix, n is the number of targets
% B: budget
% Output: 
% remove: binary vector, the targets being removed in the final solution
% fval: defender EU of the final solution

function [remove, fval] = greedy2(Params, B)
n = size(Params, 2);
t = 1;
b = 0;
counter = 1;
remove = zeros(1,n);
cov = zeros(1,n);
while b < B && t < size(cov, 2)
    cov = origami(Params);
    defEU = Params(1,:) .* cov - Params(2,:) .* (1-cov);
    defEU(~logical(cov)) = -inf;
    defEU = max(defEU);
    if cov(t) > 0
        Params = cat(2, Params(:,1:t-1), Params(:, t+1:end));
        b = b + 1;
        remove(counter) = 1;
    else
        t = t + 1;
    end
    counter = counter + 1;
end
cov = origami(Params);
defEU = Params(1,:) .* cov - Params(2,:) .* (1-cov);
defEU(~logical(cov)) = -inf;
defEU = max(defEU);
fval = defEU;

    