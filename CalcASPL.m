% Avg Shortest Path Length Calculator
function [avg_spl] = CalcASPL(Network)

A = Network;
SPL_mat = zeros(size(Network));
SPL_mat(A~=0) = 1;
idx = 1;
while(sum(sum(SPL_mat == 0)) > 0)
    idx = idx + 1;
    A = A*Network;
    SPL_mat(A~=0 & SPL_mat == 0) = idx;
end
avg_spl = 0;
count = 0;
for i=1:size(A,1)
    for j=(i+1):size(A,1)
        count = count+1;
        avg_spl = avg_spl + SPL_mat(i,j);
    end
end
avg_spl = avg_spl/count;

end