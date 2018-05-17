% ORIGAMI Algorithm [Kiekintveld et. al. 09]
% Input: 
% Params: 4xn positive payoff matrix, representing R^d, |P^d|, R^a, |P^a|. 
%
% Output: 
% coverage: coverage distribution, index is in the original (unsorted) order

function coverage = origami(Params)
n = size(Params, 2);
Params = [Params;1:n];
Params(2,:) = -Params(2,:);
Params(4,:) = -Params(4,:);
Params = sortrows(Params',3,'descend')';
coverage = zeros(1,n);
left = 1;
next = 2;
covBound = -inf;
while next <= n
    addedCov = (Params(3,next) - Params(3,:))./(Params(4,:) - Params(3,:)) - coverage;
    addedCov(next:end) = 0;
    for t=1:next-1
        if coverage(t) + addedCov(t) >= 1
            covBound = max(covBound, Params(4, t));
        end
    end
    if covBound > -inf || sum(addedCov) > left
        break;
    end
    coverage = coverage + addedCov;
    left = left - sum(addedCov);
    next = next + 1;
end
ratio = 1./(Params(3,:) - Params(4,:));
ratio(next:end) = 0;
coverage = coverage + ratio*left/sum(ratio);
for t=1:next-1
    if coverage(t) >= 1
        covBound = max(covBound, Params(4,t));
    end
end
if covBound > -inf
    coverage = (covBound - Params(3,:))/(Params(4,:) - Params(3,:));
end
index = Params(end,:);
coverage = sortrows([coverage; index]', 2, 'ascend')';
coverage = coverage(1,:);
