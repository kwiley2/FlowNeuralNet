% Clustering Coefficient Calculator
function [avg_clustering] = CalcClustering(Network)

A = Network;
avg_clustering = 0;
for u=1:size(A,1)
    indices = find(A(u,:));
    B = A(indices,indices);
    avg_clustering = avg_clustering + sum(sum(B))/((size(B,1))*(size(B,1)-1));
end
avg_clustering = avg_clustering/size(A,1);

end
