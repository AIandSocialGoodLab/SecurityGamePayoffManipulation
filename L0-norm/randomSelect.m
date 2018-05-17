% Randomized weighted average algorithm [Eppstein and Hirschberg, 1999]
% Input:
% V: vector of values
% W: vector of weights
% k: parameter (k values to remove)
% Output: 
% Astar: intermediate result, see A^* in [Eppstein and Hirschberg, 1999]
function Astar = randomSelect(V, W, k)
AL = sum(V)/sum(W);
AR = inf;
n = size(V, 2);
S = ones(1, n);
while size(V, 2) > 1
    ninit = size(V, 2);
    n = ninit;
    i = randi(n);
    delta = inf*ones(1,n);
    A = inf*ones(1,n);
    for j=1:n
        if W(i) == W(j)
            delta(j) = V(j) - V(i);
            A(j) = -inf;
        else
            delta(j) = W(i) - W(j);
            A(j) = (V(i) - V(j))/(W(i) - W(j));
        end
    end
    E = logical(delta == 0);
    X = logical(any([all([A<=AL;delta>0]); all([A>=AR;delta<0])]));
    Y = logical(any([all([A<=AL;delta<0]); all([A>=AR;delta>0])]));   
    Z = logical(~any([E;X;Y]));
    while true
        if sum(Z) > 0
            n = size(V, 2);
            Amed = median(A(Z));
            f = V - Amed * W;
            F = sum(maxk(f, n-k));
            if F == 0
                Astar = Amed;
                return
            elseif F > 0
                AL = Amed;
            else
                AR = Amed;
            end
            
            i = randi(n);
            delta = inf*ones(1,n);
            A = inf*ones(1,n);
            for j=1:n
                if W(i) == W(j)
                    delta(j) = V(j) - V(i);
                    A(j) = -inf;
                else
                    delta(j) = W(i) - W(j);
                    A(j) = (V(i) - V(j))/(W(i) - W(j));
                end
            end
            E = logical(delta == 0);
            X = logical(any([all([A<=AL;delta>0]); all([A>=AR;delta<0])]));
            Y = logical(any([all([A<=AL;delta<0]); all([A>=AR;delta>0])]));   
            Z = logical(~any([E;X;Y]));
        end
        if sum(X) + sum(E) >= n - k
            rmvEq = min(sum(E), sum(X) + sum(E) + k - n);
            Emv = logical(all([E;cumsum(E) <= rmvEq]));
            remove = logical(any([Y;Emv]));
            V = V(~remove);
            W = W(~remove);
            k = k - sum(remove);
%         elseif sum(Y) + sum(E) >= k
%             rmvEq = min(sum(E), sum(Y) + sum(E) - k);
%             Emv = logical(all([E;cumsum(E) <= rmvEq]));
%             remove = logical(any([X;Emv]));
%             newv = sum(V(remove));
%             neww = sum(W(remove));
%             V = [V(~remove), newv];
%             W = [W(~remove), neww];
        end
        if sum(Z) <= ninit/32
            break;
        end
    end
    if k == 0
        Astar = sum(V)/sum(W);
        return
    end
end
Astar = V(1)/W(1);
end
                
                
                    
                
            
            
        