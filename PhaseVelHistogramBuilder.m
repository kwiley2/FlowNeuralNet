% Phase Velocity Histogram Maker

K = [0:0.25:1]; % coupling parameter
N_avg = 10; % Number of instantiations to average over
N_neurons = 100;
edges = 0:5:60;

% Number of runs of simulation = (size(K) couplings)*(N_avg
% instantiations)*2(Kuramoto or Full)
Phase_vel_vec_kuramoto = zeros(size(K,2),N_neurons,N_avg);
Phase_vel_vec_full = zeros(size(K,2),N_neurons,N_avg);

for n=1:N_avg
    Phase_vel_vec_kuramoto(:,:,n) = FlowDiffusionNeuralSimAlt(K,1,0);
    Phase_vel_vec_full(:,:,n) = FlowDiffusionNeuralSimAlt(K,1,15e-3);
end

Phase_vel_hist_kuramoto = zeros(size(K,2),size(edges,2)-1,N_avg);
Phase_vel_hist_full = zeros(size(K,2),size(edges,2)-1,N_avg);

for n=1:N_avg
    for k=1:size(K,2)
        Phase_vel_hist_kuramoto(k,:,n) = histcounts(Phase_vel_vec_kuramoto(k,:,n),edges);
        Phase_vel_hist_full(k,:,n) = histcounts(Phase_vel_vec_full(k,:,n),edges);
    end
end

Avg_phase_vel_hist_kuramoto = 1/N_avg*sum(Phase_vel_hist_kuramoto,3);
Avg_phase_vel_hist_full = 1/N_avg*sum(Phase_vel_hist_full,3);

f = [0:1/size(K,2):(1-1/size(K,2))];
cm = colormap; % returns the current color map
colorIDs = floor((size(cm,1)-1)*f)+1;
figure(1);
hold on;
for k=1:size(K,2)
    plot([2.5:5:58.5],Avg_phase_vel_hist_kuramoto(k,:),'linestyle','-',"Color",cm(colorIDs(k),:));
    plot([2.5:5:58.5],Avg_phase_vel_hist_full(k,:),'linestyle','--',"Color",cm(colorIDs(k),:));
end
figure(1);
title("Kuramoto + Full");
hold off;