%Finite size analysis
N_avg = 5;
N = [10:10:150];
%N = 100;
K = 110./N;
Beta = [0 15e-3];
order_param_mat = zeros(7500,1);
var_vec = zeros(size(N,2),N_avg,size(Beta,2));
var_vec_avg = zeros(size(N,2),size(Beta,2));
for b=1:size(Beta,2)
    for i=1:size(N,2)
        for n=1:N_avg
            [order_param_mat,~,~,~,~,~] = FlowDiffusionNeuralSimAlt(Beta(b),K(i),0,0,0,N(i));
            var_vec(i,n,b) = var(order_param_mat);
        end
        var_vec_avg(i,b) = mean(var_vec(i,:,b),2);
    end
end

Colors = [0.8 0.2 0; 0 0.2 0.8; 0.2 0.6 0.2];
%Colors = [1 1 0.4; 1 0.5 0.6; 200/256 26/256 54/256];
%txtcol = [1 216/255 95/255];
txtcol = [0 0 0];
bkgdcol = [1 1 1];
txtsize = 14;
%txtcol = [1 216/255 95/255];
figure(1);
scatter(N,var_vec_avg(:,1),30,txtcol,'filled');
xlabel("Number of neurons");
ylabel("Variance in Order Parameter");
%title("Order Parameter Variance vs. System Size");
title("Kuramoto Model",'color',txtcol);
ylim([0,0.05]);
ax = gca
set(ax, {'XColor', 'YColor'}, {txtcol, txtcol});
set(gca,'Color',bkgdcol);
set(gcf,'Color',bkgdcol);
set(gca, 'FontSize', txtsize);

figure(2);
scatter(N,var_vec_avg(:,2),30,txtcol,'filled');
xlabel("Number of neurons");
ylabel("Variance in Order Parameter");
%title("Order Parameter Variance vs. System Size");
title("Full Model",'color',txtcol);
ylim([0,0.05]);
ax = gca
set(ax, {'XColor', 'YColor'}, {txtcol, txtcol});
set(gca,'Color',bkgdcol);
set(gcf,'Color',bkgdcol);
set(gca, 'FontSize', txtsize);