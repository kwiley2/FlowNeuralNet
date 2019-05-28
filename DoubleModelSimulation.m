% Triple model simulation (Full + Null Model Kuramoto Low Order + Null Kuramoto High Order)
Beta_Full = 15e-3;
Beta_Kuramoto = 0;
%K_coupling = [0:0.1:5 5.2:0.2:10];
K_coupling = [0:0.5:5 6:1:10];
%K_coupling = [0 1 2 5];

t_int = 30; % Length of simulation in seconds (or ms or whatever)
dt = 4e-3; % Step Size
tsteps = floor(t_int/dt); % Number of steps
t_start = 1; % When to turn on neurons
t_stable_sec = 20; % How long to average over after stability
t_stable = 20/dt;


[order_param_full,O2_vec_full,Internal_freq_vec_low,Internal_freq_vec_high,A_neural,Adj_nd] = FlowDiffusionNeuralSimAlt(Beta_Full,K_coupling,0,0,0,0);
[order_param_kuramoto_low,~,~,~,~] = FlowDiffusionNeuralSimAlt(Beta_Kuramoto,K_coupling,Internal_freq_vec_low,A_neural,Adj_nd,0);
[order_param_kuramoto_high,~,~,~,~] = FlowDiffusionNeuralSimAlt(Beta_Kuramoto,K_coupling,Internal_freq_vec_high,A_neural,Adj_nd,0);

order_param_full_mean = mean(order_param_full(:,end-t_stable:end),2);
order_param_full_std_dev = sqrt(var(order_param_full(:,end-t_stable:end)'));

order_param_kuramoto_low_mean = mean(order_param_kuramoto_low(:,end-t_stable:end),2);
order_param_kuramoto_low_std_dev = sqrt(var(order_param_kuramoto_low(:,end-t_stable:end)'));

order_param_kuramoto_high_mean = mean(order_param_kuramoto_high(:,end-t_stable:end),2);
order_param_kuramoto_high_std_dev = sqrt(var(order_param_kuramoto_high(:,end-t_stable:end)'));

F = [order_param_kuramoto_low(:,2500:7500)' order_param_full(:,2500:7500)' order_param_kuramoto_high(:,2500:7500)'];

O2_mean_full = mean(mean(O2_vec_full(:,:,end-t_stable:end),3),2);
%O2_var_full = var(mean(O2_vec_full(:,:,end-t_stable:end),3)');
O2_var_full = sum((mean(O2_vec_full(:,:,end-t_stable:end),3) - O2_mean_full).^2,2);

Colors = [0.8 0.2 0; 0 0.2 0.8; 0.2 0.6 0.2];
%Colors = [1 1 0.4; 1 0.5 0.6; 200/256 26/256 54/256];
%txtcol = [1 216/255 95/255];
txtcol = [0 0 0];
bkgdcol = [1 1 1];
fontsize = 14;
figure(1);
%boxplot(F,'Positions',[(K_coupling-0.04) (K_coupling) (K_coupling + 0.04)],'OutlierSize',1,'Colors',Colors,'Widths',0.015,'MedianStyle','target','Symbol','o');
a = boxplot(order_param_full(:,2500:7500)','Positions',K_coupling-0.015,'OutlierSize',1,'Colors',Colors(1,:),'Widths',0.015,'MedianStyle','target','Symbol','o');
hold on;
b = boxplot(order_param_kuramoto_low(1:3,2500:7500)','Positions',K_coupling(1:3)+0.045,'OutlierSize',1,'Colors',Colors(2,:),'Widths',0.015,'MedianStyle','target','Symbol','o');
c = boxplot(order_param_kuramoto_high(6:end,2500:7500)','Positions',K_coupling(6:end)+0.015,'OutlierSize',1,'Colors',Colors(3,:),'Widths',0.015,'MedianStyle','target','Symbol','o');
%for ih=1:size(order_param_full,1)
%    set(a(:,ih),'LineWidth',1.5); % Set the line width of the Box outlines here
%end
%for ih=1:7
%    set(b(:,ih),'LineWidth',1.5); % Set the line width of the Box outlines here
%end
%for ih=1:62
%    set(c(:,ih),'LineWidth',1.5); % Set the line width of the Box outlines here
%end
set(gca,'ycolor',Colors(2,:)) 
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
xticks([0 1 2 3 4 5 6 7 8 9 10]);
xticklabels([0 1 2 3 4 5 6 7 8 9 10]);
%hold on;
%plot(K_coupling,fullOrderVecMean);
text(0.25,0.9,'I','FontSize',20,'FontName','Serif','color',txtcol);
xline(0.75,'LineWidth',1.5,'Color',txtcol);
text(1.8,0.9,'II','FontSize',20,'FontName','Serif','color',txtcol);
xline(2.75,'LineWidth',1.5,'Color',txtcol);
text(7.5,0.4,'III','FontSize',20,'FontName','Serif','color',txtcol);
%xline(4.25,'LineWidth',1,'Color','k');
%text(4.55,0.4,'IV','FontSize',20,'FontName','Serif');
ylim([0 1]);
xlim([0 10]);
xlabel("Coupling Constant",'color',txtcol);
ylabel("Order Parameter",'color',txtcol);
%title("Order vs Coupling Strength");
%xline(1.4,'LineWidth',1,'Color','k');
%text(4.3, 0.75, 'sin(x)');
text(5.9, 0.18, 'o', 'Color', Colors(1,:));
text(5.9, 0.13, 'o', 'Color', Colors(2,:));
text(5.9, 0.08, 'o', 'Color', Colors(3,:));
text(6.05,0.177,'Full Model','Color',txtcol);
text(6.05,0.127,'Low Synchronization Kuramoto','color',txtcol);
text(6.05,0.077,'High Synchronization Kuramoto','color',txtcol);
ax = gca
set(gca, 'FontSize', fontsize);
set(ax, {'XColor', 'YColor'}, {txtcol, txtcol});
set(gca,'Color',bkgdcol);
set(gcf,'Color',bkgdcol);
hold off;

figure(2);
histogram(Internal_freq_vec_low,'facecolor',Colors(2,:),'facealpha',1);
ax = gca
set(ax, {'XColor', 'YColor'}, {txtcol, txtcol});
set(gca,'Color',bkgdcol);
set(gcf,'Color',bkgdcol);
set(gca, 'FontSize', fontsize);
xlabel("Resource-Modified Internal Frequency",'color',txtcol);
ylabel("Number of Oscillators",'color',txtcol);


figure(3);
histogram(Internal_freq_vec_high,'facecolor',Colors(3,:),'facealpha',1);
ax = gca
set(ax, {'XColor', 'YColor'}, {txtcol, txtcol});
set(gca,'Color',bkgdcol);
set(gcf,'Color',bkgdcol);
set(gca, 'FontSize', fontsize);
xlabel("Resource-Modified Internal Frequency",'color',txtcol);
ylabel("Number of Oscillators",'color',txtcol);

%{
figure(4);
O2_mean_vs_t = squeeze(mean(O2_vec_full,2));
plot(O2_mean_vs_t(6,1000:end),order_param_full(6,1000:end));
ax = gca
set(ax, {'XColor', 'YColor'}, {txtcol, txtcol});
set(gca,'Color',bkgdcol);
set(gcf,'Color',bkgdcol);
set(gca, 'FontSize', fontsize);
xlabel("Mean Oxygen Level",'color',txtcol);
ylabel("Order Parameter",'color',txtcol);
scatter(O2_mean_vs_t(7,1000),order_param_full(7,1000));
hold on;
figure(5);
hold on;
for i=500:(floor(size(O2_mean_vs_t,2)/2))
    figure(4);
    scatter(O2_mean_vs_t(6,4*i),order_param_full(6,4*i));
    figure(5);
    scatter(4*i,order_param_full(6,4*i));
    pause(0.001);
end
figure(4);
hold off;
figure(5);
hold off;
%}
%{
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
Diff_vec = abs(order_param_kuramoto_low_mean - order_param_full_mean);
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
rPred = (order_param_kuramoto_low_mean.^4  - 4*fit(2)*(O2_mean_full(1)./O2_mean_full)/((O2_var_vec(end)-O2_var_vec(1)))^(1).*(O2_var_vec-O2_var_vec(1)).^(1)./K_coupling_2').^(0.25);
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
plot(K_coupling,order_param_kuramoto_low_mean - rPred);
hold off;
%}