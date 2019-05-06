% Double model simulation (Full + Null Model Kuramoto)
Beta_Full = 15e-3;
Beta_Kuramoto = 0;
K_coupling = [0:0.1:5 5.2:0.2:10];

t_int = 30; % Length of simulation in seconds (or ms or whatever)
dt = 4e-3; % Step Size
tsteps = floor(t_int/dt); % Number of steps
t_start = 1; % When to turn on neurons
t_stable_sec = 20; % How long to average over after stability
t_stable = 20/dt;


[order_param_full,O2_vec_full,Internal_freq_vec,A_neural,Adj_nd] = FlowDiffusionNeuralSimAlt(Beta_Full,K_coupling,0,0,0);
[order_param_kuramoto,~,~,~,~] = FlowDiffusionNeuralSimAlt(Beta_Kuramoto,K_coupling,Internal_freq_vec,A_neural,Adj_nd);

order_param_full_mean = mean(order_param_full(:,end-t_stable:end),2);
order_param_full_std_dev = sqrt(var(order_param_full(:,end-t_stable:end)'));

order_param_kuramoto_mean = mean(order_param_kuramoto(:,end-t_stable:end),2);
order_param_kuramoto_std_dev = sqrt(var(order_param_kuramoto(:,end-t_stable:end)'));

F = [order_param_kuramoto(:,2500:7500)' order_param_full(:,2500:7500)'];

O2_mean_full = mean(mean(O2_vec_full(:,:,end-t_stable:end),3),2);
%O2_var_full = var(mean(O2_vec_full(:,:,end-t_stable:end),3)');
O2_var_full = sum((mean(O2_vec_full(:,:,end-t_stable:end),3) - O2_mean_full).^2,2);

Colors = [0.8 0.2 0; 0 0.2 0.8];
figure(1);
boxplot(F,'Positions',[(K_coupling-0.015) (K_coupling+0.015)],'OutlierSize',1,'Colors',Colors,'Widths',0.015,'MedianStyle','target','Symbol','o');
set(gca,'ycolor',Colors(2,:)) 
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
xticks([0 1 2 3 4 5]);
xticklabels([0 1 2 3 4 5]);
hold on;
%plot(K_coupling,fullOrderVecMean);
%text(0.25,0.9,'I','FontSize',20,'FontName','Serif');
%xline(0.55,'LineWidth',1,'Color','k');
%text(0.63,0.9,'II','FontSize',20,'FontName','Serif');
%xline(0.85,'LineWidth',1,'Color','k');
%text(2.5,0.4,'III','FontSize',20,'FontName','Serif');
%xline(4.25,'LineWidth',1,'Color','k');
%text(4.55,0.4,'IV','FontSize',20,'FontName','Serif');
ylim([0 1]);
xlabel("Coupling Constant");
ylabel("Order Parameter");
title("Order vs Coupling Strength");
%xline(1.4,'LineWidth',1,'Color','k');
hold off;

figure(2);
scatter(K_coupling,O2_var_full,'o','MarkerEdgeColor',[0 0.4 0]);
%text(0.25,3,'I','FontSize',20,'FontName','Serif');
%xline(0.55,'LineWidth',1,'Color','k');
%text(0.63,3,'II','FontSize',20,'FontName','Serif');
%xline(0.85,'LineWidth',1,'Color','k');
%text(2.5,0.4,'III','FontSize',20,'FontName','Serif');
%xline(4.25,'LineWidth',1,'Color','k');
%text(4.55,0.4,'IV','FontSize',20,'FontName','Serif');
ylabel("O2 Variance");
set(gca,'ycolor',[0 0.4 0]);
title("O2 Variance vs Coupling Strength");
xlabel("Coupling Constant");
ylim([0 4.5]);
%xline(1.4,'LineWidth',1,'Color','k');

figure(3);
Diff_vec = abs(order_param_kuramoto_mean - order_param_full_mean);
Ind_var = K_coupling.^(-2);
Ind_var = Ind_var';
fit = [ones(51,1) Ind_var(end-50:end)]\Diff_vec((end-50):end)
scatter(Ind_var(1:end),Diff_vec(1:end));
hold on;
plot([0:0.01:1],(fit(1)+fit(2)*[0:0.01:1]));
xlim([0 0.2]);
hold off;

%Calculate prediction based off of energy theory
%alpha = 7.06e5;
K_coupling_2 = K_coupling.*K_coupling;
%rPred = (mean(KuramotoOrderVec.order_param_vec,2).^4 - 4.3*4*alpha/(100^2*400)*(O2_var_vec-O2_var_vec(1))./K_coupling_2').^(0.25);
rPred = (order_param_kuramoto_mean.^4  - 4*fit(2)*(O2_mean_full(1)./O2_mean_full)/((O2_var_vec(end)-O2_var_vec(1)))^(1).*(O2_var_vec-O2_var_vec(1)).^(1)./K_coupling_2').^(0.25);
%rPred = (1 - 4*alpha/(100^2*387.12)*(O2_var_vec - O2_var_vec(1))./K_coupling_2').^(0.25);
rPred = real(rPred) - imag(rPred);
rPred(rPred<0.001) = 0;
rPred(isnan(rPred)) = 0;
figure(1);
hold on;
plot(K_coupling,rPred);
hold off;

figure(5);
plot(K_coupling,Diff_vec);
hold on;
plot(K_coupling,order_param_kuramoto_mean - rPred);
hold off;