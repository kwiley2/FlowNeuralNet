function [adj,Comms] = blockmodel_2community_network(N1,N2,p_in,p_out) 
N = N1+N2;
adj = rand(N,N); %Generate random matrix
adj = adj - diag(diag(adj)); %delete self loops

%Now we want to keep given entries if their values are less that p_in or
%p_out (depending on their location in the adjacency matrix). Since our
%adjacency matrix is unweighted, we can do this with simple rounding to 1
%or 0.

%within communities
adj(1:N1,1:N1) = floor(adj(1:N1,1:N1)+p_in);
adj((N1+1):N,(N1+1):N) = floor(adj((N1+1):N,(N1+1):N)+p_in);

%between communities
adj(1:N1,(N1+1):N) = floor(adj(1:N1,(N1+1):N)+p_out);
adj((N1+1):N,1:N1) = floor(adj((N1+1):N,1:N1)+p_out);

%symmetrize (without affecting probabilities)
adj = adj - tril(adj);
adj = adj + adj';

Comms = [ones(1,N1),2*ones(1,N-N1-1)];
