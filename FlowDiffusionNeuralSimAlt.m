% Simulate Flow + Diffusion dynamics

%Interesting params {K_input, Beta, tanh_spread, O2_avg,D}
% {2, 1e-6, 10, 1,0.5} - Similar bursting behavior to that seen before


%function [avg_order_param,var_order_param,avg_O2,var_O2,] = FlowDiffusionNeuralSim(K_inpu[t,Beta,tanh_spread,O2_avg,D)
% Want to run the full model simulation, then run a Kuramoto simulation
% with the same topology and internal frequency distribution as the full
% model simulation has at maximum coupling. Fingers crossed that this null
% model will give the best comparison with the energy picture
function [order_param_vec,O2_diff_vec,Internal_freq_vec_low,Internal_freq_vec_high,A_neural,Adj_nd] = FlowDiffusionNeuralSimAlt(Beta,K_coupling,KuramotoFreqs,Kuramoto_A_neural,Kuramoto_Adj_nd,N_neur)
%function [order_param_vec,O2_diff_vec,Internal_freq_vec,A_neural,Adj_nd] = FlowDiffusionNeuralSimAlt(Beta,K_coupling,N_neur)
%Beta = 15e-3;
%K_coupling = 1.1;
%N_neur = 100;
%When running DoubleModelSimulation, set N_neur to 0
%When running FiniteSizeAnalysis, N_neur will never be 0
if(N_neur == 0)
    N_neurons = 100;
else
    N_neurons = N_neur;
end

%K_coupling = [0:5]*1;
%K_coupling = [1.3 1.4];
K_independent = 1; % 0 if varying D,Beta; 1 if varying K
%if(K_independent == 1)
    %K_coupling = [0:0.1:5 5.2:0.2:10];
    %K_coupling = [0 1 2];
%    K_coupling = 0;
%else
%    K_coupling = 5;
%end
Scales = 1+0.1*[-5:1:5];
%Beta = 0;
%Beta = 15e-3; %conversion between phase speed and O2 usage
tanh_spread = 10; % a factor to control how narrow the tanh distribution is
O2_avg = 0.85; %defines how far the tanh function is offset
%D = 40;
D = 5;
animated = 0;


%% Variable Initialization
% Load in all necessary information from the flow network generation
Network = load('SampleFlowNetwork');
delta_x = Network.FlowNetwork.LatticeDeltaX;
delta_y = Network.FlowNetwork.LatticeDeltaY;
Diffusion_Lattice = Network.FlowNetwork.LatticeAdj;
Flow_Lattice = Network.FlowNetwork.ConductanceNetwork;
%Flow_Lattice = sparse(Flow_Lattice);
Current_Source = Network.FlowNetwork.CurrentSource;
Current_Sinks = Network.FlowNetwork.CurrentSinks';
Nodes_X = Network.FlowNetwork.X_positions;
Nodes_Y = Network.FlowNetwork.Y_positions;
N_nodes = size(Diffusion_Lattice,1);

% Parameters I really want to vary are the strength of the
% neural coupling and the O2 usage defined by Beta
% It could also be interesting to vary the shift in the tanh argument
% defined by O2_avg and the neural topology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Free Simulation Parameters (Other parameters, such as the underlying
% lattice structure and flow network details exist, but are altered in
% the piece of code which generates the flow network)

t_int = 30; % Length of simulation in seconds (or ms or whatever)
dt = 4e-3; % Step Size
tsteps = floor(t_int/dt); % Number of steps
t_start = 1; % When to turn on neurons
t_stable_sec = 20; % How long to average over after stability
t_stable = 20/dt;

% Amount of O2 that enters flow layer at source node per time step
O2_bath = 1; %if I consider the flow layer static, this is the O2 supply to every sink node at each time step
I_O2 = size(Current_Sinks,1)*O2_bath;


% Diffusion Coefficients
D_flow_to_diff = D;
D_diff = D;
D_diff_to_neural = D;

K = 1.7e-4; %Conversion between synaptic weights and currents
%N_neurons = N_neur;% number of neurons
%p = 0.9*1/sqrt(N_neurons); %probability of a given synapse existing (ER)
p = 0.2;
N1 = N_neurons/2; %size of community 1
N2 = N_neurons-N1; %size of community 2
p_in = 0.9*1/sqrt(N_neurons); %in-community probability of synapse existing (Blockmodel)
p_out = 0.2*p_in; %out-of-community probability of synapse existing
rho = 1.4; % Needed to generate E-R neural net graph
Mean_Resistance = 500; %resistances ~500 Ohms
Var_Resistance = 100;
Mean_Capacitance = 20e-6; %capacitances ~20uF
Var_Capacitance = 3e-6;
Mean_Freq = 1/(Mean_Resistance*Mean_Capacitance);
Var_Freq = Mean_Freq*sqrt((Var_Resistance/Mean_Resistance)^2+(Var_Capacitance/Mean_Capacitance)^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Generate Neural Adjacency Matrix

%Simplified Kuramoto model version of ER network
%{
A_neural = gen_adj(N_neurons, p, rho); %ER network
A_neural(eye(size(A_neural,1))==1) = 0; %Delete self-loops
A_neural = abs(A_neural);
A_neural = triu(A_neural);
A_neural = A_neural+A_neural';
A_neural(A_neural~=0) = 1;
A_neural = sparse(A_neural);
%}

% Watts-Strogatz Small World Network
A_neural = WattsStrogatz(N_neurons,(ceil((N_neurons-1)*p/2)),1);

% REPLACE THESE LINES LATER
if(Beta==0 && N_neur == 0)
    A_neural = Kuramoto_A_neural;
end
% REPLACE THESE LINES LATER


% Generate symmetric 2 community network
%[A_neural,Comms] = blockmodel_2community_network(N1,N2,p_in,p_out);


% Input and Output Current at each node (not currents along edges, just
% what enters and leaves the system)
I_sink_source = zeros(N_nodes,1);
I_sink_source(Current_Source) = 1;
I_sink_source(Current_Sinks(:)) = -1/size(Current_Sinks,1);

% Find current flow through each edge given a steady-state flow
% This may change if we later allow the neurons to modify flow from steady-
% state, and this is why we designed the flow network as we did
Edge_Currents = zeros(N_nodes);
G = -Flow_Lattice + eye(N_nodes).*sum(Flow_Lattice,2);
Voltages = pinv(G)*I_sink_source;
for i=1:N_nodes
    for j=1:N_nodes
        Edge_Currents(i,j) = Flow_Lattice(i,j)*(Voltages(i)-Voltages(j));
    end
end
Transition_matrix = Edge_Currents;
Transition_matrix(Transition_matrix<0) = 0;
Transition_matrix_sum = sum(Transition_matrix,2);
Transition_matrix_sum(Transition_matrix_sum==0) = 1;
% If we have a current sink, increase the denominator so that the
% probability ISN'T 1, indicating current leaving the system.
Transition_matrix_sum(Current_Sinks(:)) = Transition_matrix_sum(Current_Sinks(:)) + 1/size(Current_Sinks,1);
for i=1:size(Transition_matrix,1)
    for j=1:size(Transition_matrix,2)
        Transition_matrix(i,j) = Transition_matrix(i,j)/Transition_matrix_sum(i);
    end
end     

%Initialize O2 content of each node in the flow layer, as well as the
%amount of O2 available for diffusion into the diffusion layer
Node_O2_Flow = zeros(N_nodes,tsteps+1);
Diffusion = zeros(size(Current_Sinks,1),tsteps+1);
Node_O2_Flow(Current_Source) = I_O2;

%Initialize the O2 content of each node in the Diffusion layer
Node_O2_Diff = zeros(N_nodes,tsteps);
if(Beta==0)
    Node_O2_Diff(:,1) = 1;
end

%Initialize data variables for the neural layer
Voltages = zeros(N_neurons,tsteps);
Phases = zeros(N_neurons,tsteps);
Phase_vel = zeros(N_neurons,tsteps);
Phases(:,t_start) = 2*pi*rand(N_neurons,1);
Voltage_thresh = normrnd(20e-3,1e-3,[N_neurons,1]);
Active = ones([N_neurons,tsteps]);
O2 = zeros(N_neurons,tsteps);
if(Beta==0)
    O2(:,1) = 1;
end

% Define variable electrical properties for each neuron (assuming normal
% distribution)
%{
Capacitances = normrnd(Mean_Capacitance,Var_Capacitance,[N_neurons,1]);
Resistances = normrnd(Mean_Resistance,Var_Resistance,[N_neurons,1]);
Natural_Freqs = zeros(N_neurons,1);
for i=1:N_neurons
    Natural_Freqs(i) = 1/(Capacitances(i)*Resistances(i)); %doesn't really make sense as a frequency, but it does define a time scale, so sort of close
end
%}
% Alternate Gaussian distribution
Natural_Freqs = normrnd(Mean_Freq,Var_Freq,[N_neurons,1]);

% Rescale natural frequencies for comparison to standard Kuramoto
if(Beta==0)
    % Hardcoded numbers for the moment, but could change later
    %Natural_Freqs = 1/1.9*normrnd(KuramotoMean,KuramotoStdDev,[N_neurons,1]);
    if(N_neur == 0)
        Natural_Freqs = KuramotoFreqs/1.9;
    else
        Natural_Freqs = Natural_Freqs/1.9;
    end
    %Natural_Freqs = Natural_Freqs/1.9;
end

% Decide which nodes in the diffusion layer each neuron will be connected
% to
Random_List = randperm(size(Diffusion_Lattice,1));
Adj_nd = zeros(N_neurons,size(Node_O2_Diff,1));
for i=1:N_neurons
    Adj_nd(i,Random_List(i)) = 1;
end
Adj_nd = sparse(Adj_nd);
if(Beta==0 && N_neur == 0)
    Adj_nd = Kuramoto_Adj_nd;
end
% REPLACE THESE LINES ^^^
Flow_Lattice = sparse(Flow_Lattice);
Diffusion_Lattice = sparse(Diffusion_Lattice);

order_param_vec = zeros(size(K_coupling,2),tsteps);
order_param_vec_mean = zeros(size(K_coupling,2),1);
order_param_vec_var = zeros(size(K_coupling,2),1);

O2_diff_vec = zeros(size(K_coupling,2),size(Diffusion_Lattice,1),tsteps);
O2_diff_vec_mean = zeros(size(K_coupling,2),size(Diffusion_Lattice,1));
O2_diff_mean = zeros(size(K_coupling,2),1);
O2_diff_var = zeros(size(K_coupling,2),1);

O2_neur_vec = zeros(size(K_coupling,2),N_neurons,tsteps);
O2_neur_vec_mean = zeros(size(K_coupling,2),N_neurons);
O2_neur_mean = zeros(size(K_coupling,2),1);
O2_neur_var = zeros(size(K_coupling,2),1);

Phase_vel_full_vec = zeros(size(K_coupling,2),N_neurons);
%% Update loop
% Scale adjacency matrices once for timesave
iter = 0;
%K fluctuating
if(K_independent == 1)
    iter = size(K_coupling,2);
%D and Beta fluctuating
else
    iter = size(Scales,2);
end

for k=1:iter
    %K fluctuating
    if(K_independent == 1)
        K = K_coupling(k);
        Adj_nd_scaled = D_diff_to_neural*Adj_nd;
        Diffusion_Lattice_scaled = D_diff*Diffusion_Lattice;
    end
    
    %D and Beta fluctuating
    if(K_independent == 0)
        K = K_coupling;
        Beta_mod = Scales(k)*Beta;
        D_mod = Scales(k)*D;
        Adj_nd_scaled = D_mod*Adj_nd;
        Diffusion_Lattice_scaled = D_mod*Diffusion_Lattice;
    end
    
    A_neural_scaled = K*A_neural;
    
    for i=1:(tsteps-1)
        %Version with K fluctuating
        if(K_independent==1)
            [Phases(:,i+1), O2(:,i+1), Node_O2_Diff(:,i+1),Phase_vel(:,i+1)] = update_system(Phases(:,i), O2(:,i), Natural_Freqs, A_neural_scaled, dt, K,Node_O2_Diff(:,i),Adj_nd_scaled,D_diff_to_neural,O2_avg,Beta,tanh_spread,Diffusion_Lattice_scaled,Current_Sinks,D_diff,D_flow_to_diff,O2_bath);
        %Version with D and Beta fluctuating
        else
            [Phases(:,i+1), O2(:,i+1), Node_O2_Diff(:,i+1),Phase_vel(:,i+1)] = update_system(Phases(:,i), O2(:,i), Natural_Freqs, A_neural_scaled, dt, K,Node_O2_Diff(:,i),Adj_nd_scaled,D_mod,O2_avg,Beta_mod,tanh_spread,Diffusion_Lattice_scaled,Current_Sinks,D_mod,D_mod,O2_bath);
        end
    end
    order_param = zeros(tsteps,1);
    for t=1:tsteps
        for i=1:N_neurons
            order_param(t) = order_param(t) + exp(1j*Phases(i,t));
        end
    end
    %order_param(t) = order_param_vec(k)/N_neurons;
    order_param_vec(k,:) = abs(order_param)/N_neurons;
    order_param_vec_mean(k) = mean(abs(order_param((end-t_stable):end)))/N_neurons;
    order_param_vec_var(k) = var(abs(order_param((end-t_stable):end)))/N_neurons^2;
    O2_diff_vec(k,:,:) = Node_O2_Diff;
    O2_neur_vec(k,:,:) = O2;
    Phase_vel_full_vec(k,:) = mean(Phase_vel(:,(end-t_stable):end),2);
    Phases(:,1) = Phases(:,tsteps);
    O2(:,1) = O2(:,tsteps);
    Node_O2_Diff(:,1) = Node_O2_Diff(:,tsteps);
    Phase_vel(:,1) = Phase_vel(:,tsteps);
end

%{
figure(1);
s1 = scatter(K_coupling,order_param_vec_mean(1:size(K_coupling,2)));
hold on;
%s2 = scatter(fliplr(K_coupling(1:(end-1))),order_param_vec_mean((size(K_coupling,2)+1):end));
ylim([0,1.15])
plot(K_coupling,ones(size(K_coupling)));
%legend([s1,s2],{'K increasing','K decreasing'});
hold off;

figure(2);
s2 = scatter(K_coupling,order_param_vec_var(1:size(K_coupling,2)));
hold on;
%s4 = scatter(fliplr(K_coupling(1:(end-1))),order_param_vec_var((size(K_coupling,2)+1):end));
%ylim([0,1.15])
%plot(K_coupling,ones(size(K_coupling)));
%legend([s3,s4],{'K increasing','K decreasing'});
hold off;

figure(3);
e1 = errorbar(K_coupling,order_param_vec_mean(1:size(K_coupling,2)),sqrt(order_param_vec_var(1:size(K_coupling,2))),'.');
%}



%% Calc Numeric Output


order_param = zeros(tsteps,1);
for t=1:tsteps
    for i=1:N_neurons
        order_param(t) = order_param(t) + exp(1j*Phases(i,t));
    end
end
group_phase = angle(order_param);
group_phase = group_phase + pi;
order_param = abs(order_param)/N_neurons;

sys_order_param = calc_n_step_order_param_vs_t(Phases,A_neural,20);
comm1_order_param = calc_n_step_order_param_vs_t(Phases(1:N1,:),A_neural(1:N1,1:N1),20);
comm2_order_param = calc_n_step_order_param_vs_t(Phases((N1+1):N_neurons,:),A_neural((N1+1):N_neurons,(N1+1):N_neurons),20);
one_step_order_param = calc_n_step_order_param_vs_t(Phases,A_neural,1);
one_step_avg_order_param = mean(one_step_order_param(:,(end-t_stable):end),2);
sys_avg_order_param = mean(sys_order_param((end-t_stable):end));
comm1_avg_order_param = mean(comm1_order_param((end-t_stable):end));
comm2_avg_order_param = mean(comm2_order_param((end-t_stable):end));
sys_var_order_param = var(sys_order_param((end-t_stable):end));
Node_O2_time_avg = 1/t_stable*(sum(Node_O2_Diff(:,(end-t_stable):end),2));
avg_O2 = mean(Node_O2_time_avg);
var_O2 = var(Node_O2_time_avg);
avg_vel = mean(Phase_vel,1);

Phase_Diffs = bsxfun(@minus,Phases,group_phase');
Avg_Phase_Diffs = mean(abs(sin(Phase_Diffs(:,(end-1000):end))),2);
edges = 0:5:60;
Phase_vel_out = Phase_vel_full_vec;
%}

%% Plots
%{


figure(2);
plot(Avg_Phase_Diffs);
is_phase_outlier = zeros(N_neurons,1);
is_vel_outlier = zeros(N_neurons,1);
for i=1:size(Avg_Phase_Diffs)
    if(Avg_Phase_Diffs(i) > 0.4)
        text(i,Avg_Phase_Diffs(i),int2str(i));
        is_phase_outlier(i) = 1;
    end
end
avg_phase_vel = mean(mean(Phase_vel(:,(end-2000):end)));
sample_vel_outlier = 0;
vel_outlier_thresh = 0.1; %0.05 in synchronized regime
for i=1:size(Avg_Phase_Diffs,1)
    if(abs(mean(Phase_vel(i,(end-2000):end),2)-avg_phase_vel)/avg_phase_vel>vel_outlier_thresh)
        is_vel_outlier(i) = 1;
        text(i,(Avg_Phase_Diffs(i)*1.05),"vel");
        sample_vel_outlier = i;
    end
end 
title("Large Phase Offset and Asynchronous Neurons");
xlabel("Neuron number");
ylabel("Distance from mean");
%}

%{
figure(1);
t = dt*(1:tsteps);
yyaxis left
ylabel("Phase velocity");
plot(t,avg_vel);
yyaxis right
ylabel("Order parameter");
plot(t,sys_order_param(1,:));
xlabel("time");
title("Phase Velocity and Order Parameter");
xlim([0 t_int]);
%}
%{
figure(2);
if(K_independent == 1)
    boxplot(order_param_vec','labels',K_coupling,'Positions',K_coupling);
else
    boxplot(order_param_vec','labels',Scales,'Positions',Scales);
end
xticks([0 1 2 3 4 5]);
xticklabels([0 1 2 3 4 5]);
xlabel("Coupling Constant");
ylabel("Order Parameter");
title("Order vs Coupling Strength");
%}
O2_diff_vec_mean = mean(O2_diff_vec(:,:,(end-3000):end),3);
O2_diff_mean = mean(O2_diff_vec_mean,2);
O2_diff_var = sum((O2_diff_vec_mean - O2_diff_mean).^2,2);

plot(tanh(10*(O2_diff_vec_mean(:,:)-O2_avg))+1);

O2_neur_vec_mean = mean(O2_neur_vec(:,:,(end-3000):end),3);
O2_neur_mean = mean(O2_neur_vec_mean,2);
O2_neur_var = sum((O2_neur_vec_mean - O2_neur_mean).^2,2);

%Internal_freq_vec = Natural_Freqs.*(tanh(tanh_spread*(mean(O2(:,end-t_stable:end),2) - O2_avg))+1);
Internal_freq_vec_low = Natural_Freqs.*(tanh(tanh_spread*(O2_neur_vec_mean(1,:)' - O2_avg))+1);
Internal_freq_vec_high = Natural_Freqs.*(tanh(tanh_spread*(O2_neur_vec_mean(end,:)' - O2_avg))+1);
%{
figure(3);
histogram(mean(Phase_vel(:,(end-t_stable):end),2));
mean(mean(Phase_vel(:,(end-t_stable):end),2))
sqrt(var(mean(Phase_vel(:,(end-t_stable):end),2)))
%}

%{
figure(2);
plot(one_step_avg_order_param);
refline(0,sys_avg_order_param);
ylim([0,1]);

figure(3);
plot(t,sys_order_param(1,:),t,one_step_order_param(1,:));

figure(4);
avg_vel_1 = mean(Phase_vel(1:N1,:),1);
avg_vel_2 = mean(Phase_vel((N1+1):N_neurons,:),1);
plot(t,avg_vel_1,t,avg_vel_2);

figure(5);
plot(t,comm1_order_param(1,:),t,comm2_order_param(1,:));
%}

%{
K_coupling;
avg_order_param;

figure(2);
plot(Phases(1,1:end));

figure(1);
plot(order_param);
figure(3);
plot(Phase_vel(1,1:end));

figure(4);
P_fft = fft(Phases(1,:)-pi);
P2 = abs(P_fft/tsteps);
P1 = P2(1:tsteps/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = (1/dt)*(0:(tsteps/2))/tsteps;
plot(f*2*pi,P1);
xlim([0,100]);

figure(5);
dP_fft = fft(Phase_vel(1,:));
dP2 = abs(dP_fft/tsteps);
dP1 = dP2(1:tsteps/2+1);
dP1(2:end-1) = 2*dP1(2:end-1);
df = (1/dt)*(0:(tsteps/2))/tsteps;
plot(df,dP1);
xlim([0,10]);



figure(6);
Phase_vel_vec = zeros(size(Phase_vel,1)*size(Phase_vel,2),1);
Phase_vel_time = zeros(size(Phase_vel,1)*size(Phase_vel,2),1);
for i=1:size(Phase_vel,1)
    for j=1:size(Phase_vel,2)
        Phase_vel_vec(i+(j-1)*size(Phase_vel,1)) = Phase_vel(i,j);
        Phase_vel_time(i+(j-1)*size(Phase_vel,1)) = j;
    end
end
Phase_vel_hist = [Phase_vel_vec,Phase_vel_time];
counts = hist3(Phase_vel_hist,[200,tsteps]);
imshow(counts);
axis on;



figure(7);
imagesc(O2);
colorbar;
%}

% iMac Display
%{
set(1,'Position',[0,1000,600,300]);
set(2,'Position',[0,600,600,300]);
set(3,'Position',[0,200,600,300]);
set(4,'Position',[700,1000,600,300]);
set(5,'Position',[700,600,600,300]);
%set(6,'Position',[1400,1000,600,300]);
%}


% Macbook Air Display
%{
set(1,'Position',[700,600,600,300]);
set(2,'Position',[0,600,600,300]);
set(3,'Position',[0,100,600,300]);
%set(4,'Position',[700,1000,600,300]);
%set(5,'Position',[700,600,600,300]);
set(6,'Position',[0,100,1200,400]);
set(7,'Position',[700,100,600,300]);
%}
%{
figure(1);
plot(order_param)

figure(2);
plot1 = imagesc(Node_O2_Diff);
plot1(1,1).CData(Current_Sinks,1:5) = 255;
%plot1(1,1).CData(deg<3,6:10) = 200;


Diffusion_Graph = graph(Diffusion_Lattice);
Diffusion_Graph.Nodes.Size = Node_O2_Diff(:,end-1);
title('Diffusion O_2 Content vs. Time');
xlabel('Time (ms)');
ylabel('Node');
colorbar;
%}

%%%%%%%%%%%%%%%%%%%%% Cool Slider Time Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
figure(3);
cm = colormap;
Flow_Graph = graph(Flow_Lattice);
Diffusion_Graph = graph(Diffusion_Lattice);
Diffusion_Graph.Nodes.Size = Node_O2_Diff(:,end-1);
plot2 = plot(Flow_Graph,'XData',Nodes_X,'YData',Nodes_Y,'LineWidth',20*Flow_Graph.Edges.Weight/max(Flow_Graph.Edges.Weight));
is_doubly_connected = zeros(N_neurons,1);
for i=1:N_nodes
    f = Diffusion_Graph.Nodes.Size(i)/max(Diffusion_Graph.Nodes.Size);
    colorID = sum(f>[0:1/length(cm(:,1)):1]);
    highlight(plot2,i,'NodeColor',cm(colorID,:));
    highlight(plot2,i,'MarkerSize',15);
end
for j=1:N_neurons
    highlight(plot2,Random_List(j),'Marker','s');
    labelnode(plot2,Random_List(j),j);
end
Markers = get(plot2,'Marker');
for j=1:size(Current_Sinks,1)
    if(Markers{Current_Sinks(j)} == "square")
        highlight(plot2,Current_Sinks(j),'Marker','p');
    else
        highlight(plot2,Current_Sinks(j),'Marker','d');
    end
end 
Markers = get(plot2,'Marker');
doubles = find(Markers=="pentagram");
for i=1:size(doubles,2)
    is_doubly_connected(find(Random_List==doubles(i))) = 1;
end

colorbar;
title('Diffusion Layer (Square = Neuron Cxn, Diamond = O2 Source, Star = Both, Circle = Neither)');
h = uicontrol('style','slider','units','pixel','position',[20 20 300 20]);
addlistener(h,'ContinuousValueChange',@(hObject, event) updatePlot(hObject, event,x,plot2,tsteps,Node_O2_Diff));

fprintf('Number of Outliers: %d\n',sum(is_phase_outlier));
fprintf('Doubly Connected Outliers: %d\n',sum(is_doubly_connected(find(is_phase_outlier))));
%}

%{
figure(4);
Neural_Graph = graph(A_neural);
plot(Neural_Graph);
%}

%finding how many "steps" away from a source each neuron is
%{
Oxygen_vector = zeros(size(Adj_nd,2),1);
Oxygen_vector(Current_Sinks) = 1;
Steps_vec = zeros(size(Adj_nd,2),1);
step_ctr = 1;
Steps_vec(Oxygen_vector>0) = step_ctr;
Diag_mat = eye(size(Diffusion_Lattice,1)).*sum(Diffusion_Lattice,2);
while(sum(Oxygen_vector==0)>0)
    Oxygen_vector_temp = Oxygen_vector + 0.01*(Diffusion_Lattice - Diag_mat)*Oxygen_vector;
    step_ctr = step_ctr + 1;
    Steps_vec(((Oxygen_vector_temp>0)-(Oxygen_vector>0))==1) = step_ctr;
    Oxygen_vector = Oxygen_vector_temp;
end
Neuron_steps = Adj_nd*Steps_vec;
figure(8);
scatter(Neuron_steps,O2(:,end));
coeffs = polyfit(Neuron_steps,O2(:,end),1);
fittedX = linspace(min(Neuron_steps), max(Neuron_steps), 200);
fittedY = polyval(coeffs,fittedX);
hold on;
plot(fittedX,fittedY);
hold off;
title("Distance from Source vs. Equilibrium O2");
xlabel("Number of Steps");
ylabel("Neuron Equilibrium O2");
%Calculate R^2
SS_tot = 0;
SS_res = 0;
O2_bar = mean(O2(:,end));
for i=1:size(O2(:,end),1)
    SS_tot = SS_tot + (O2(i,end) - O2_bar)^2;
    SS_res = SS_res + (O2(i,end) - (coeffs(2) + coeffs(1)*Neuron_steps(i)))^2;
end
R2 = 1 - SS_res/SS_tot;
%}

%{
figure(9);
if(sample_vel_outlier==0)
    sample_vel_outlier=1;
end
vel = Phase_vel(sample_vel_outlier,2000:end);
group_phase = angle(sum(exp(1i*Phases(:,2000:end)),1));
sin_phase_diff = sin(Phases(sample_vel_outlier,2000:end)-group_phase);
diff_O2 = Node_O2_Diff(find(Adj_nd(sample_vel_outlier,:)),2000:(end-1));
neur_O2 = O2(sample_vel_outlier,2000:end);
intrinsic_vel = Natural_Freqs(sample_vel_outlier)*(tanh(tanh_spread*(neur_O2-O2_avg))+1);
if(animated)
    for i = 1:length(vel)
        subplot(3,1,1);
        xlim([0,40]);
        plot(vel(1:i),sin_phase_diff(1:i),'Color','k');
        hold on;
        plot(vel(i),sin_phase_diff(i),'Marker','o','MarkerEdgeColor','r');
        hold off;
        subplot(3,1,2);
        xlim([0,40]);
        plot(vel(1:i),diff_O2(1:i),'Color','k');
        hold on;
        plot(vel(i),diff_O2(i),'Marker','o','MarkerEdgeColor','r');
        hold off;
        subplot(3,1,3);
        xlim([0,40]);
        plot(vel(1:i),intrinsic_vel(1:i),'Color','k');
        hold on;
        plot(vel(i),intrinsic_vel(i),'Marker','o','MarkerEdgeColor','r');
        hold off;
        pause(0.0001);
    end
else
    subplot(3,1,1);
    xlim([0,40]);
    plot(vel,sin_phase_diff);
    subplot(3,1,2);
    xlim([0,40]);
    plot(vel,diff_O2);
    subplot(3,1,3);
    xlim([0,40]);
    plot(vel,intrinsic_vel);
end
%}

%{
figure(10);
plot(mean(Phase_vel(:,1000:end),1),order_param(1000:end));
%}

%{
figure(11);
x = sym('x',[270 1]);
Source_vec = zeros(size(Flow_Lattice,1),1);
Source_vec(Current_Sinks) = 1;
Sink_vec = Adj_nd'*ones(N_neurons,1);
diff_deg_vec = sum(Diffusion_Lattice,2);
solx = solve(D*O2_bath*Source_vec - D*Source_vec.*x + D*Diffusion_Lattice*x-D*diff_deg_vec.*x - Beta*avg_phase_vel*Sink_vec == 0,x);
fields = fieldnames(solx);
O2_pred_synch = zeros(270,1);
for i=1:numel(fields)
    O2_pred_synch(i) = solx.(fields{i});
end
Diffusion_Graph2 = graph(Diffusion_Lattice);
Diffusion_Graph2.Nodes.Size = O2_pred_synch;
plot3 = plot(Flow_Graph,'XData',Nodes_X,'YData',Nodes_Y,'LineWidth',20*Flow_Graph.Edges.Weight/max(Flow_Graph.Edges.Weight));
for i=1:N_nodes
    f = Diffusion_Graph2.Nodes.Size(i)/max(Diffusion_Graph2.Nodes.Size);
    colorID2 = sum(f>[0:1/length(cm(:,1)):1]);
    highlight(plot3,i,'NodeColor',cm(colorID2,:));
    highlight(plot3,i,'MarkerSize',15);
end
for j=1:N_neurons
    highlight(plot3,Random_List(j),'Marker','s');
    labelnode(plot3,Random_List(j),j);
end
Markers = get(plot3,'Marker');
for j=1:size(Current_Sinks,1)
    if(Markers{Current_Sinks(j)} == "square")
        highlight(plot3,Current_Sinks(j),'Marker','p');
    else
        highlight(plot3,Current_Sinks(j),'Marker','d');
    end
end 
colorbar;
%}

%{
figure(5);
predicted_avg_vel = D*(O2_bath-O2_avg+1/tanh_spread)/(Beta*(1+N_neurons/size(Current_Sinks,1))+D/tanh_spread/mean(Natural_Freqs));
avg_vel_vec = predicted_avg_vel*ones(size(Natural_Freqs,1),1);
O2_pred_neur = Adj_nd*O2_pred_synch - Beta/D*avg_phase_vel;
internal_vel_vec = Natural_Freqs.*(tanh(tanh_spread*(O2_pred_neur-O2_avg))+1);
diff_vec = abs(avg_vel_vec-internal_vel_vec);
degree_vec = sum(A_neural,2);
thresh_vec = K_coupling*degree_vec*order_param(end);
plot(diff_vec-thresh_vec);
zero_x = linspace(min(diff_vec), max(diff_vec), 200);
zero_y = zeros(size(zero_x,1),1);
title("Predicting Asynchronous Neurons");
xlabel("Neuron Number");
ylabel("Prediction (>0 means can't synch)");
hold on;
plot(zero_x,zero_y);
hold off;
%}

%{
figure(13);
x_coords = linspace(0,1,1000);
x_coords = x_coords';
y_coords_pred = zeros(size(x_coords,1),1);
y_coords_data_based = zeros(size(x_coords,1),1);
predicted_phase_vel = D*(O2_bath-O2_avg+1/tanh_spread)/(Beta*(1+N_neurons/size(Current_Sinks,1))+D/(tanh_spread*mean(Natural_Freqs)));
for i=1:size(x_coords,1)
    y_coords_pred(i) = calcSqrtSum(N_neurons,Natural_Freqs,K_coupling,degree_vec,x_coords(i),tanh_spread,O2_pred_neur,O2_avg,predicted_phase_vel);
    y_coords_data_based(i) =calcSqrtSum(N_neurons,Natural_Freqs,K_coupling,degree_vec,x_coords(i),tanh_spread,max(O2(:,(end-2000):end),[],2),O2_avg,mean(mean(Phase_vel(:,(end-2000):end))));
end
plot(x_coords,y_coords_pred);
axis([0 1 0 1]);
hold on;
plot(x_coords,x_coords);
plot(x_coords,y_coords_data_based);
hold off;
%}

%{
syms r
r_pred = vpasolve(r - calcSqrtSum(N_neurons,Natural_Freqs,K_coupling,degree_vec,r,tanh_spread,O2_pred_neur,O2_avg,predicted_phase_vel) == 0,r,[0.1 1]);
r_pred_data_based = vpasolve(r - calcSqrtSum(N_neurons,Natural_Freqs,K_coupling,degree_vec,r,tanh_spread,max(O2(:,(end-2000):end),[],2),O2_avg,mean(mean(Phase_vel(:,(end-2000):end)))) == 0,r,[0.1 1]);
%}

%{
figure(14);
P_fft = fft(Phases(1,:)-pi);
P2 = abs(P_fft/tsteps);
P1 = P2(1:tsteps/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = (1/dt)*(0:(tsteps/2))/tsteps;
plot(f*2*pi,P1);
xlim([0,100]);
%}

%{
figure(14);
P_fft = fft(order_param(2500:end));
P2 = abs(P_fft/(tsteps-2500));
P1 = P2(1:(0.5*(tsteps-2500)+1));
P1(2:end-1) = 2*P1(2:end-1);
f = (1/dt)*(0:(tsteps-2500)/2)/(tsteps-2500);
plot(f(2:end),P1(2:end));
xlim([0,10]);
[~,max_idx] = max(P1(2:end)); %ignores 0 bin, which is the mean velocity
f(max_idx)
%O2_pred_rand = zeros(270,1);
%y = sym('y',[370 1]); %Sym variable for all nodes (1:270 are diffusion layer, 271:370 are neural layer)
%[soly, solz] = solve([(O2_bath-y).*Source_vec + Diffusion_Lattice*y - y.*sum(Diffusion_Lattice,2) + Adj_nd'*z - y.*sum(Adj_nd',2) == 0,Adj_nd*y - z == Beta*Natural_Freqs/D.*(tanh(tanh_spread*(z-O2_avg))+1)],[y z]);
%}

%{
figure(4);
Flow_Graph = graph(Flow_Lattice);
plot3 = plot(Flow_Graph,'XData',Nodes_X,'YData',Nodes_Y,'LineWidth',20*Flow_Graph.Edges.Weight/max(Flow_Graph.Edges.Weight));
highlight(plot3,Current_Sinks,'NodeColor','r');
title('Flow Layer (Edge Size = Conductivity, Green = Source, Red = Sink)');
%}
%{
% Calculate Some Things for the Neural Layer plots
% Net Activity of All Neurons
Net_Activity = sum(Fired,1);
% DFT of Net Activity (maybe has meaning, though looks like noise to me)
y_net = fft(Net_Activity);
f_net = (0:length(y_net)-1)*50/length(y_net);
y_1 = fft(Fired(1,:));
f_1 = (0:length(y_1)-1)*50/length(y_1);

figure(4);
imagesc(Fired);
title('Action Potential History of Neurons (Dark Blue = Nothing, Yellow = Fired');
xlabel('Time (ms)');
ylabel('Neuron Label');

figure(5);
plot(Net_Activity);
xlim([0 tsteps]);
title('Net Activity of All Neurons vs. Time');
xlabel('Time (ms)');
ylabel('Number of Fired Action Potentials');


figure(6);
plot(f_net(2:floor(end/2)),abs(y_net(2:floor(end/2))));
xlim([0 5]);
title('DFT of Net Activity');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
%}
%set(1,'Position',[0,1000,600,300]);
%set(2,'Position',[300,127,300,300]);
%set(3,'Position',[0,128,300,300]);
%set(4,'Position',[600,1000,800,300]);
%{
set(5,'Position',[600,300,800,150]);
set(6,'Position',[600,90,800,150]);
%}
%}
end

function [Node_O2_next,diffusable_O2] = update_flow(Node_O2_current,Transition_matrix,Current_Source,Current_Sinks,I_O2)

Node_O2_next = Node_O2_current;
%Discrete version
%{
diffusable_O2 = zeros(size(Current_Sinks,1),1);
for i=1:size(Transition_matrix,1)
    N_quanta = Node_O2_current(i);
    p_vec = Transition_matrix(i,:);
    p_vec = p_vec';
    [O2_dist,diff] = dist_quanta(N_quanta,p_vec);
    for j=1:size(Current_Sinks,1)
        if(i==Current_Sinks(j))
            diffusable_O2(j) = diff;
        end
    end
    Node_O2_next = Node_O2_next + O2_dist;
end
Node_O2_next(Current_Source) = I_O2;
%}

%Continuous version
for i=1:size(Current_Sinks,1)
    if(Node_O2_next(Current_Sinks(i))>0)
        Node_O2_next(Current_Sinks(i)) = Node_O2_next(Current_Sinks(i)) - I_O2/size(Current_Sinks,1);
    end
end
Node_O2_next = (Transition_matrix')*Node_O2_next;
Node_O2_next(Current_Source) = I_O2;
diffusable_O2(:) = I_O2/size(Current_Sinks,1);

end

function Node_O2_next = update_diff(Node_O2_current,Diffusion_Lattice,Diffusable,Current_Sinks,dt,D_diff,D_transfer)

Node_O2_next = Node_O2_current;
Node_O2_difference = Node_O2_current-Node_O2_current';
Node_O2_transfer_vec = sum(Diffusion_Lattice'.*Node_O2_difference,1)';
Node_O2_next = Node_O2_current + dt*D_diff*Node_O2_transfer_vec;

Diffusable_vec = zeros(size(Node_O2_current,1),1);
Diffusable_vec(Current_Sinks) = Diffusable;

diffuse = Node_O2_current < Diffusable_vec;
Node_O2_next = Node_O2_next + dt*D_transfer*(Diffusable_vec - Node_O2_current).*diffuse;

%Discrete Diffusion
%{
for i=1:size(Transition_matrix,1)
    % Diffusion process
    N_quanta = Node_O2_current(i);
    p_vec = Transition_matrix(i,:);
    p_vec = p_vec';
    O2_dist = dist_quanta(N_quanta,p_vec);
    % If we're updating a sink node, and the diffusion layer has less O2
    % than the flow layer, then probabilistically distribute the difference
    % to the diffusion layer.
    if(sum(Current_Sinks==i)>0)
        if(Node_O2_current(i) < Diffusable(find(Current_Sinks==i)))
            N_transfer_quanta = Diffusable(find(Current_Sinks==i))-Node_O2_current(i);
            rand_vec = rand(N_transfer_quanta,1);
            transferred = sum(rand_vec < p_transfer);
            Node_O2_next(i) = Node_O2_next(i) + transferred;
        end
    end    
    
    Node_O2_next = Node_O2_next + O2_dist;
end
%}

end

function [Active_next, Fired_next, Voltages_next, O2_next, Diff_O2_next] = update_neural_if(Active_current, Fired_current, Voltages_current, O2_current, Resistances, Capacitances, A_neural, dt, K, Noise_variance, Voltage_thresh,Node_O2_diff,Diff_to_Neural_Cxns,D_diff_to_neural,O2_need) 
% This just preallocates size. The values change in a way that doesn't rely
% on the current state, so the specific values here don't matter
Voltages_next = Voltages_current;
O2_next = O2_current;
Active_next = Active_current;
Fired_next = zeros(size(Fired_current,1),1);
Diff_O2_next = Node_O2_diff;

Fire_mat = eye(size(Fired_current,1)).*Fired_current;
I_fire = K*Fire_mat*A_neural;
N_neurons = size(Voltages_current,1);

for j=1:N_neurons
    Voltages_next(j) = Voltages_current(j) + (normrnd(0,Noise_variance)/Capacitances(j)-Voltages_current(j)/(Capacitances(j)*Resistances(j)))*dt + dt*sum(I_fire(:,j))/Capacitances(j);
    if(O2_current(j) < 0)
        % Pull oxygen diffusively from the connected node in the diffusion
        % layer
        O2_next(j) = O2_current(j) + D_diff_to_neural*Node_O2_diff(Diff_to_Neural_Cxns(j))*dt;
        Diff_O2_next(Diff_to_Neural_Cxns(j)) = Diff_O2_next(Diff_to_Neural_Cxns(j)) - D_diff_to_neural*Node_O2_diff(Diff_to_Neural_Cxns(j))*dt;
        if(O2_next(j) >= 0)
            Active_next(j) = 1;
        end
        if(O2_next(j) < 0)
            Active_next(j) = 0;
        end
    end
    if(Voltages_next(j) > Voltage_thresh(j) && Active_next(j)==1)
        Fired_next(j) = 1;
        Voltages_next(j) = 0;
        O2_next(j) = -O2_need;
        Active_next(j) = 0;
    end
end
end

function [Phase_next, O2_next, Diff_O2_next,Phase_vel] = update_neural_kuramoto(Phase_current, O2_current, Natural_Freqs, A_neural, dt, K,Node_O2_diff,Adj_nd,D_nd,O2_avg,Beta,tanh_spread)
% This just preallocates size. The values change in a way that doesn't rely
% on the current state, so the specific values here don't matter
Phase_next = Phase_current;
O2_next = O2_current;
Diff_O2_next = Node_O2_diff;
O2_rest = 0;
%alpha = O2_avg;

N_neurons = size(Phase_current,1);
%Sin_matrix = sin(Phase_current-Phase_current');
%Interaction_vec = sum(A_neural'.*Sin_matrix,1)';

O2_diff_nd = Node_O2_diff-O2_current';
Interlayer_O2_vec = sum(Adj_nd'.*O2_diff_nd,1)';

%tanh_O2 = tanh(tanh_spread*(O2_current - O2_avg));
%phi_dot = Natural_Freqs.*(tanh_O2+1) + K*Interaction_vec;
phi_dot = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current,O2_avg,Phase_current,A_neural,K);
%O2_neur_dot_vec = calc_O2_neur_dots(Node_O2_diff,O2_current,Adj_nd,D_nd,phi_dot,Beta);
Phase_next = Phase_current + dt*phi_dot;
%Fired = Phase_next>(2*pi);
for j=1:N_neurons
    while(Phase_next(j)>2*pi)
        Phase_next(j) = Phase_next(j)-2*pi;
    end
end
O2_next = O2_current + dt*D_nd*Interlayer_O2_vec - Beta*phi_dot;
Diff_O2_next = Node_O2_diff - dt*D_nd*(Adj_nd'*Interlayer_O2_vec);

Phase_vel = phi_dot;
end

function [phi_dot_vec] = calc_phi_dots(Natural_Freqs,S,O2_neur_vec,O2_avg,Phases,A_neural_scaled,K_coupling) 
Sin_matrix = sin(Phases-Phases');
Interaction_vec = sum(A_neural_scaled.*Sin_matrix',2);
tanh_O2 = tanh(S*(O2_neur_vec - O2_avg));
phi_dot_vec = Natural_Freqs.*(tanh_O2+1) + Interaction_vec;
end

function [O2_neur_dot_vec] = calc_O2_neur_dots(O2_diff_vec,O2_neur_vec,Adj_nd_scaled,D_nd,phi_dot_vec,Beta,Natural_Freqs)

O2_neur_dot_vec = sum(Adj_nd_scaled.*(O2_diff_vec-O2_neur_vec')',2) - Beta*phi_dot_vec;

end

function [O2_diff_dot_vec] = calc_O2_diff_dots(O2_diff_vec,O2_bath,Current_Sinks,Adj_diff_scaled,Adj_nd_scaled,O2_neur_vec,D_nd,D_diff,D_fd)
O2_bath_vec = O2_diff_vec;
O2_bath_vec(Current_Sinks) = O2_bath;
O2_diff_dot_vec = D_fd*(O2_bath_vec - O2_diff_vec) - sum(Adj_diff_scaled.*(O2_diff_vec - O2_diff_vec'),2) - sum(Adj_nd_scaled'.*(O2_diff_vec - O2_neur_vec'),2);
end

function [Phase_next, O2_next, Diff_O2_next,Phase_vel] = update_system(Phase_current, O2_current, Natural_Freqs, A_neural_scaled, dt, K,Node_O2_diff,Adj_nd_scaled,D_nd,O2_avg,Beta,tanh_spread,Diffusion_Lattice_scaled,Current_Sinks,D_diff,D_transfer,O2_bath)
%{
% Euler Method
phi_dot_vec = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current,O2_avg,Phase_current,A_neural,K);
O2_diff_dot_vec = calc_O2_diff_dots(Node_O2_diff,O2_bath,Current_Sinks,Diffusion_Lattice,Adj_nd,O2_current,D_nd,D_diff,D_transfer);
O2_neur_dot_vec = calc_O2_neur_dots(Node_O2_diff,O2_current,Adj_nd,D_nd,phi_dot_vec,Beta);

Phase_next = wrapTo2Pi(Phase_current + dt*phi_dot_vec);
O2_next = O2_current + dt*O2_neur_dot_vec;
Diff_O2_next = Node_O2_diff + dt*O2_diff_dot_vec;
Phase_vel = phi_dot_vec;
%}
% Runge-Kutta Method (k = phi, l = O2_diff, m = O2_neur)
phi_dot_vec_0 = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current,O2_avg,Phase_current,A_neural_scaled,K);
k0 = dt*phi_dot_vec_0;
l0 = dt*calc_O2_diff_dots(Node_O2_diff,O2_bath,Current_Sinks,Diffusion_Lattice_scaled,Adj_nd_scaled,O2_current,D_nd,D_diff,D_transfer);
m0 = dt*calc_O2_neur_dots(Node_O2_diff,O2_current,Adj_nd_scaled,D_nd,phi_dot_vec_0,Beta,Natural_Freqs);

phi_dot_vec_1 = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current+0.5*m0,O2_avg,Phase_current+0.5*k0,A_neural_scaled,K);
k1 = dt*phi_dot_vec_1;
l1 = dt*calc_O2_diff_dots(Node_O2_diff+0.5*l0,O2_bath,Current_Sinks,Diffusion_Lattice_scaled,Adj_nd_scaled,O2_current+0.5*m0,D_nd,D_diff,D_transfer);
m1 = dt*calc_O2_neur_dots(Node_O2_diff+0.5*l0,O2_current+0.5*m0,Adj_nd_scaled,D_nd,phi_dot_vec_1,Beta,Natural_Freqs);

phi_dot_vec_2 = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current+0.5*m1,O2_avg,Phase_current+0.5*k1,A_neural_scaled,K);
k2 = dt*phi_dot_vec_2;
l2 = dt*calc_O2_diff_dots(Node_O2_diff+0.5*l1,O2_bath,Current_Sinks,Diffusion_Lattice_scaled,Adj_nd_scaled,O2_current+0.5*m1,D_nd,D_diff,D_transfer);
m2 = dt*calc_O2_neur_dots(Node_O2_diff+0.5*l1,O2_current+0.5*m1,Adj_nd_scaled,D_nd,phi_dot_vec_2,Beta,Natural_Freqs);

phi_dot_vec_3 = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current+m2,O2_avg,Phase_current+k2,A_neural_scaled,K);
k3 = dt*phi_dot_vec_3;
l3 = dt*calc_O2_diff_dots(Node_O2_diff+l2,O2_bath,Current_Sinks,Diffusion_Lattice_scaled,Adj_nd_scaled,O2_current+m2,D_nd,D_diff,D_transfer);
m3 = dt*calc_O2_neur_dots(Node_O2_diff+l2,O2_current+m2,Adj_nd_scaled,D_nd,phi_dot_vec_3,Beta,Natural_Freqs);

Phase_next = wrapTo2Pi(Phase_current+1/6*(k0+2*k1+2*k2+k3));
O2_next = O2_current+1/6*(m0+2*m1+2*m2+m3);
Diff_O2_next = Node_O2_diff+1/6*(l0+2*l1+2*l2+l3);
Phase_vel = 1/6*(phi_dot_vec_0+2*phi_dot_vec_1+2*phi_dot_vec_2+phi_dot_vec_3);
end


function [output] = dphase_dt(natural_freq, K, interaction,alpha,O2_rest,O2_current) 
output = (natural_freq+K*interaction)*exp(-alpha*(O2_rest-O2_current));
end

function [n_step_order_param] = calc_n_step_order_param_vs_t(Phases,Adj_mat,n)
pre_screen = zeros(size(Adj_mat));
for k=0:n
    pre_screen = pre_screen + Adj_mat^k;
end
pre_screen = pre_screen + pre_screen';
screen = (pre_screen>0);
order_param = screen*exp(1j*Phases);
N_vec = sum(screen,2);
order_param_mag = abs(order_param);
n_step_order_param = (1./N_vec).*order_param_mag;
end

function updatePlot(hObject,event,x,hplot,tsteps,Node_O2_Diff)
t = floor((tsteps-1)*get(hObject,'Value'))+1;
Weights = Node_O2_Diff(:,t);
cm = colormap;
for i=1:size(Node_O2_Diff,1)
    f = Weights(i)/max(Weights);
    colorID = sum(f>[0:1/length(cm(:,1)):1]);    
    highlight(hplot,i,'NodeColor',cm(colorID,:));
end
drawnow;
end