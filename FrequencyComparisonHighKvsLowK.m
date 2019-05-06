% Comparing low coupling to high coupling internal frequency distribution
N_neurons = 100;
N_trials = 40;
%edges = 0:10:160;
edges = 0:3:60;

Internal_freq_vec = zeros(N_neurons,N_trials);
for i=1:N_trials
    [~,~,Internal_freq_vec(:,i),~,~] = FlowDiffusionNeuralSimAlt(15e-3,0,0,0,0);
end

Internal_freq_hist = zeros(N_trials,size(edges,2)-1);

for n=1:N_trials
    Internal_freq_hist(n,:) = histcounts(Internal_freq_vec(:,n),edges);
end

average_hist = 1/N_trials*sum(Internal_freq_hist,1);

scatter(1.5:3:58.5,(average_hist));
xlim([0 60]);
