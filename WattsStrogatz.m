% Small-world network generator
% First generate a ring network with node degree 2*k
% Then, rewire each edge at random with probability p
% As always, assume that the network is undirected and no self-loops
function [Small_World_Network] = WattsStrogatz(N_nodes,K_nearest_neighbors,p_rewiring)

K = K_nearest_neighbors;
N = N_nodes;
p = p_rewiring;

% Generate the initial Ring Lattice
Small_World_Network = zeros(N,N);
for k=1:K
    kth_connections = eye(N,N);
    cycle_idx = [(N-k+1):N 1:(N-k)];
    kth_connections = kth_connections(cycle_idx,:);
    kth_connections = kth_connections + kth_connections';
    Small_World_Network = Small_World_Network + kth_connections;
end

% Loop through edges and rewire probabilistically
for i=1:N
    for j=i:i+K
        if(rand() < p)
            rewired = 0;
            while(rewired == 0)
                rand_idx = randi([1 N]);
                if((rand_idx ~= i) && (Small_World_Network(i,rand_idx) == 0))
                    Small_World_Network(i,rand_idx) = 1;
                    Small_World_Network(rand_idx,i) = 1;
                    if(j > 100)
                        Small_World_Network(i,mod(j,N)) = 0;
                        Small_World_Network(mod(j,N),i) = 0;
                    else
                        Small_World_Network(i,j) = 0;
                        Small_World_Network(j,i) = 0;
                    end
                    rewired = 1;
                end
            end
        end
    end
end

end