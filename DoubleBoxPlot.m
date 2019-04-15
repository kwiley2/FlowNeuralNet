% Double Box Plot

%K_coupling = [0:0.1:2 2.5:0.5:5];
K_coupling = [0:0.1:5 5.2:0.2:10];
KuramotoOrderVec = load('BoxPlotKuramotoAlt.mat','order_param_vec');
FullModelOrderVec = load('BoxPlotFullModelAlt.mat','order_param_vec');
FullModelO2VarVec = load('BoxPlotFullModelAlt.mat','O2_diff_var');

F = [KuramotoOrderVec.order_param_vec(:,2500:7500)' FullModelOrderVec.order_param_vec(:,2500:7500)'];
O2_var_vec = FullModelO2VarVec.O2_diff_var;
Colors = [0.8 0.2 0; 0 0.2 0.8];
fullOrderVecMean = max(FullModelOrderVec.order_param_vec');
%Green: 0.16 0.5 0.16
figure(1);
%boxplot(KuramotoOrderVec.order_param_vec','labels',K_coupling,'Positions',K_coupling,'Color','k','PlotStyle','compact','OutlierSize',2);
%hold on;
%boxplot(FullModelOrderVec.order_param_vec','labels',K_coupling,'Positions',K_coupling,'Color','b','PlotStyle','compact','OutlierSize',2);
boxplot(F,'Positions',[(K_coupling-0.015) (K_coupling+0.015)],'OutlierSize',1,'Colors',Colors,'Widths',0.015,'MedianStyle','target','Symbol','o');
set(gca,'ycolor',Colors(2,:)) 
set(findobj(gcf,'LineStyle','--'),'LineStyle','-')
xticks([0 1 2 3 4 5]);
xticklabels([0 1 2 3 4 5]);
hold on;
%plot(K_coupling,fullOrderVecMean);
text(0.25,0.9,'I','FontSize',20,'FontName','Serif');
xline(0.55,'LineWidth',1,'Color','k');
text(0.63,0.9,'II','FontSize',20,'FontName','Serif');
xline(0.85,'LineWidth',1,'Color','k');
text(2.5,0.4,'III','FontSize',20,'FontName','Serif');
xline(4.25,'LineWidth',1,'Color','k');
text(4.55,0.4,'IV','FontSize',20,'FontName','Serif');
ylim([0 1]);
xlabel("Coupling Constant");
ylabel("Order Parameter");
title("Order vs Coupling Strength");
xline(1.4,'LineWidth',1,'Color','k');
scatter(K_coupling,fullOrderVecMean);
hold off;

figure(2);
scatter(K_coupling,O2_var_vec,'o','MarkerEdgeColor',[0 0.4 0]);
hold on;
scatter(K_coupling,O2_var_vec,'.','MarkerEdgeColor',[0 0.4 0]);
text(0.25,3,'I','FontSize',20,'FontName','Serif');
xline(0.55,'LineWidth',1,'Color','k');
text(0.63,3,'II','FontSize',20,'FontName','Serif');
xline(0.85,'LineWidth',1,'Color','k');
text(2.5,0.4,'III','FontSize',20,'FontName','Serif');
xline(4.25,'LineWidth',1,'Color','k');
text(4.55,0.4,'IV','FontSize',20,'FontName','Serif');
ylabel("O2 Variance");
set(gca,'ycolor',[0 0.4 0]);
title("O2 Variance vs Coupling Strength");
xlabel("Coupling Constant");
ylim([0 4.5]);
xline(1.4,'LineWidth',1,'Color','k');
hold off;

figure(3);
KuramotoMeans = mean(KuramotoOrderVec.order_param_vec');
FullModelMeans = mean(FullModelOrderVec.order_param_vec');
KuramotoVars = var(KuramotoOrderVec.order_param_vec');
FullModelVars = var(FullModelOrderVec.order_param_vec');
DifferenceStdDev = sqrt(KuramotoVars + FullModelVars);
errorbar(K_coupling.^(-2),(abs(KuramotoMeans-FullModelMeans)),DifferenceStdDev);
xlim([0 0.6]);
ylim([0 1]);
Diff_vec = abs(KuramotoMeans - FullModelMeans);
Diff_vec = Diff_vec';
Ind_var = K_coupling.^(-2);
Ind_var = Ind_var';
fit = [ones(41,1) Ind_var(end-40:end)]\Diff_vec((end-40):end)
hold on;
plot([0:0.01:1],(fit(1)+fit(2)*[0:0.01:1]));
hold off;

%Calculate prediction based off of energy theory
alpha = 7.06e5;
K_coupling_2 = K_coupling.*K_coupling;
%rPred = (mean(KuramotoOrderVec.order_param_vec,2).^4 - 4.3*4*alpha/(100^2*400)*(O2_var_vec-O2_var_vec(1))./K_coupling_2').^(0.25);
rPred = (mean(KuramotoOrderVec.order_param_vec,2).^4 - 4*fit(2)/((O2_var_vec(end)-O2_var_vec(1)))^(2)*(O2_var_vec-O2_var_vec(1)).^(2)./K_coupling_2').^(0.25);
%rPred = (1 - 4*alpha/(100^2*387.12)*(O2_var_vec - O2_var_vec(1))./K_coupling_2').^(0.25);
rPred = real(rPred) - imag(rPred);
rPred(rPred<0.001) = 0;
rPred(isnan(rPred)) = 0;
figure(1);
hold on;
plot(K_coupling,rPred);
hold off;

