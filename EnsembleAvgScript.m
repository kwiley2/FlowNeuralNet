%Average over many (random) topologies
n_trials = 10;
n_couplings = 10;
ensemble_avg_order_params = zeros(n_couplings,1);
ensemble_std_dev_order_params = zeros(n_couplings,1);
for j=1:n_couplings
    K_coupling = j/100;

    avg_order_param = zeros(n_trials,1);
    for i=1:n_trials
        avg_order_param(i) = FlowDiffusionNeuralSim(K_coupling);
    end
    ensemble_avg_order_params(j) = mean(avg_order_param);
    ensemble_std_dev_order_params(j) = std(avg_order_param);
end
plot(ensemble_avg_order_params);
%ensemble_std_dev_order_param