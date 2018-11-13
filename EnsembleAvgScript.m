%Average over many (random) topologies
n_trials = 5;
n_couplings = 10;
n_D = 10;
ensemble_avg_order_params = zeros(n_couplings,n_D,1);
ensemble_std_dev_order_params = zeros(n_couplings,n_D,1);
ensemble_avg_var_order_param = zeros(n_couplings,n_D,1);
ensemble_std_dev_var_order_param = zeros(n_couplings,n_D,1);
ensemble_avg_avg_O2 = zeros(n_couplings,n_D,1);
ensemble_std_dev_avg_O2 = zeros(n_couplings,n_D,1);
ensemble_avg_var_O2 = zeros(n_couplings,n_D,1);
ensemble_std_dev_var_O2 = zeros(n_couplings,n_D,1);
for i=1:n_D
    for j=1:n_couplings
        K_coupling = j/5;
        D = 0.1*i;
        %S = 1*i;

        avg_order_param = zeros(n_trials,1);
        var_order_param = zeros(n_trials,1);
        avg_O2 = zeros(n_trials,1);
        var_O2 = zeros(n_trials,1);
        for k=1:n_trials
            [avg_order_param(k),var_order_param(k),avg_O2(k),var_O2(k)] = FlowDiffusionNeuralSimAlt(K_coupling,1e-2,10,0.95,D);
        end
        ensemble_avg_order_params(j,i) = mean(avg_order_param);
        ensemble_std_dev_order_params(j,i) = std(avg_order_param);
        ensemble_avg_var_order_param(j,i) = mean(var_order_param);
        ensemble_std_dev_var_order_param(j,i) = std(var_order_param);
        ensemble_avg_avg_O2(j,i) = mean(avg_O2);
        ensemble_std_dev_avg_O2(j,i) = std(avg_O2);
        ensemble_avg_var_O2(j,i) = mean(var_O2);
        ensemble_std_dev_var_O2(j,i) = std(var_O2);
    end
end
%{
for l=1:n_D
    figure(l);
    plot(ensemble_avg_order_params(:,l));
    title(['Avg Order Param, Diffusion Constant = ',num2str((l+3)*0.1)]);
    
    figure(l+n_D);
    plot(ensemble_avg_var_order_param(:,l));
    title(['Var Order Param, Diffusion Constant = ',num2str((l+3)*0.1)]);
    
    figure(l+2*n_D);
    plot(ensemble_avg_avg_O2(:,l));
    title(['Avg O2, Diffusion Constant = ',num2str((l+3)*0.1)]);
    
    figure(l+3*n_D);
    plot(ensemble_avg_var_O2(:,l));
    title(['Avg O2, Diffusion Constant = ',num2str((l+3)*0.1)]);
end
%}
%{
for l=1:n_O2_alpha
    figure(l);
    plot(ensemble_avg_order_params(:,l));
end
%}
%ensemble_std_dev_order_param