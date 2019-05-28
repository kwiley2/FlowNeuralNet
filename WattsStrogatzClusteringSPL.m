% Plot Clustering vs Shortest Path Length
N = 100;
K = 5;
P = [0 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1 1];
N_avg = 5;
ASPL = zeros(size(P,2),1);
ACC = zeros(size(P,2),1);
for i=1:size(P,2)
    for j=1:N_avg
        Mat = WattsStrogatz(N,K,P(i));
        ASPL(i) = ASPL(i) + CalcASPL(Mat);
        ACC(i) = ACC(i) + CalcClustering(Mat);
    end
    ASPL(i) = ASPL(i)/N_avg;
    ACC(i) = ACC(i)/N_avg;
end
figure(1);
semilogx(P,ASPL/ASPL(1),P,ACC/ACC(1));
title("Avg Shortest Path and Avg Clustering");
legend("Shortest Path","Clustering");
%figure(2);
%scatter(log(P),ACC/ACC(1));
%title("Avg Clustering");