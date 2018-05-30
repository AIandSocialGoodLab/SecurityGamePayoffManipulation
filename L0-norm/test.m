%% Main experiments for all 4 algorithms
addpath('C:\Program Files\IBM\ILOG\CPLEX_Studio128\cplex\matlab\x64_win64');
numExp = 22; % number of trials for each case
numN = 5;  % different number of targets n
testn = linspace(50, 250, numN);

milpTime = zeros(numN,numExp,2);
polyrTime = zeros(numN,numExp,2);
greedyTime = zeros(numN,numExp,2);
greedy2Time = zeros(numN,numExp,2);

fvalGreedy = -inf*ones(numN,numExp,2);
fvalGreedy2 = -inf*ones(numN,numExp,2);
fvalMILP = -inf*ones(numN,numExp,2);
fvalPoly = -inf*ones(numN,numExp,2);

for i=1:numExp
    for nt=1:numN
        save('test0201new');
        n = testn(nt);
        B = n/2;  % budget
        m = 1;  % defensive resources
        rng(i*n);
        Params = randi(2*n,4,n);
        Params = sortrows(Params',2,'descend')';
        tic;
        [finalRemove, finalZero, fvalGreedy(nt,i,1)] = greedy(Params, B);
        greedyTime(nt, i,1) = toc;
        tic;
        [remove, fvalGreedy2(nt,i,1)] = greedy2(Params, B);
        greedy2Time(nt, i,1) = toc;
        Params = sortrows(Params',3,'descend')';
        tic;
        [RemoveMILP, ZeroMILP, fvalMILP(nt,i,1),x] = oneMILP(n, Params, B, m);
        milpTime(nt, i,1) = toc;
        tic;
        [candidates, fvalPoly(nt,i,1)] = polyTime(n, Params, B, m);
        polyrTime(nt, i,1) = toc;
        disp(['n = ', num2str(n), ', instance i = ', num2str(i), ', m = 1',...
            ', Greedy time = ', num2str(greedyTime(nt,i,1)), ', Greedy2 time = ', num2str(greedy2Time(nt,i,1)),...
            ', MILP time = ', num2str(milpTime(nt,i,1)), ', random time = ', num2str(polyrTime(nt,i,1))]);
        disp(['n = ', num2str(n), ', instance i = ', num2str(i), ', m = 1',...
            ', Greedy fval = ', num2str(fvalGreedy(nt,i,1)), ', Greedy2 fval = ', num2str(fvalGreedy2(nt,i,1)),...
            ', MILP fval = ', num2str(fvalMILP(nt,i,1)), ', random fval = ', num2str(fvalPoly(nt,i,1))]);
        
        m = n/10;  % defensive resources
        Params = sortrows(Params',2,'descend')';
        tic;
        [finalRemove, finalZero, fvalGreedy(nt,i,2)] = greedy(Params, B);
        greedyTime(nt, i,2) = toc;
        tic;
        [remove, fvalGreedy2(nt,i,2)] = greedy2(Params, B);
        greedy2Time(nt, i,2) = toc;
        Params = sortrows(Params',3,'descend')';
        tic;
        [RemoveMILP, ZeroMILP, fvalMILP(nt,i,2),x] = oneMILP(n, Params, B, m);
        milpTime(nt, i,2) = toc;
        tic;
        [candidates, fvalPoly(nt,i,2)] = polyTime(n, Params, B, m);
        polyrTime(nt, i,2) = toc;
        disp(['n = ', num2str(n), ', instance i = ', num2str(i), ', m = n/10',...
            ', Greedy time = ', num2str(greedyTime(nt,i,2)), ', Greedy2 time = ', num2str(greedy2Time(nt,i,2)),...
            ', MILP time = ', num2str(milpTime(nt,i,2)), ', random time = ', num2str(polyrTime(nt,i,2))]);
        disp(['n = ', num2str(n), ', instance i = ', num2str(i), ', m = n/10',...
            ', Greedy fval = ', num2str(fvalGreedy(nt,i,2)), ', Greedy2 fval = ', num2str(fvalGreedy2(nt,i,2)),...
            ', MILP fval = ', num2str(fvalMILP(nt,i,2)), ', random fval = ', num2str(fvalPoly(nt,i,2))]);
    end
end

%%
mean(milpTime(:,1:i-1,2), 2)
mean(polyrTime(:,1:i-1,2), 2)
mean(greedyTime(:,1:i-1,2), 2)
mean(greedy2Time(:,1:i-1,2), 2)
std(milpTime(:,1:i-1,2),0, 2)
std(polyrTime(:,1:i-1,2),0, 2)
std(greedyTime(:,1:i-1,2),0, 2)
std(greedy2Time(:,1:i-1,2),0, 2)


%% solution quality of the greedy algorithms
greedyQuality = -inf * ones(5,22,2);
greedy2Quality = -inf * ones(5,22,2);
for i=1:22
    for nt=1:5
        n = nt*50;
        rng(i*n);
        Params = randi(2*n,4,n);
        Params(2,:) = -Params(2,:);
        P = min(Params(2,:));
        greedyQuality(nt,i,1) = (fvalGreedy(nt,i,1) - P) / (fvalPoly(nt,i,1) - P);
        greedy2Quality(nt,i,1) = (fvalGreedy2(nt,i,1) - P) / (fvalPoly(nt,i,1) - P);
        greedyQuality(nt,i,2) = (fvalGreedy(nt,i,2) - P) / (fvalPoly(nt,i,2) - P);
        greedy2Quality(nt,i,2) = (fvalGreedy2(nt,i,2) - P) / (fvalPoly(nt,i,2) - P);
    end
end