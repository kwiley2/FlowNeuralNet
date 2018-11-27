% Simulate Flow + Diffusion dynamics

%Interesting params {K_input, Beta, tanh_spread, O2_avg,D}
% {2, 1e-6, 10, 1,0.5} - Similar bursting behavior to that seen before


function [avg_order_param,var_order_param,avg_O2,var_O2] = FlowDiffusionNeuralSim(K_input,Beta,tanh_spread,O2_avg,D)
%{
K_input = 0.2;
Beta = 0; %conversion between phase speed and O2 usage
tanh_spread = 10; % a factor to control how narrow the tanh distribution is
O2_avg = 1; %defines how far the tanh function is offset
D = 0;
%}

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

t_int = 100; % Length of simulation in seconds (or ms or whatever)
dt = 1e-1; % Step Size
tsteps = floor(t_int/dt); % Number of steps
t_start = 1; % When to turn on neurons
t_stable = 200; % How long to average over after stability

% Amount of O2 that enters flow layer at source node per time step
O2_bath = 1; %if I consider the flow layer static, this is the O2 supply to every sink node at each time step
I_O2 = size(Current_Sinks,1)*O2_bath;


% Diffusion Coefficients (if continuous)
%D = 0.5;
D_flow_to_diff = D;
D_diff = D;
D_diff_to_neural = D;

%Transition Probabilities (if discrete)
%p_transfer = 0.5; % Probability to move from flow to diffusion layer
%p_stay = 0.5; % Probability to stay still in diffusion layer
%p_move = 1-p_stay; % Probability to move to new node in diffusion layer

K = 1.7e-4; %Conversion between synaptic weights and currents
N_neurons = 100;% number of neurons
p = 0.9*1/sqrt(N_neurons); %probability of a given synapse existing
rho = 1.4; % Needed to generate E-R neural net graph
Mean_Resistance = 500; %resistances ~500 Ohms
Var_Resistance = 100;
Mean_Capacitance = 20e-6; %capacitances ~20uF
Var_Capacitance = 3e-6;
%O2_consumption = 30; % Amount of O2 necessary to transition back into active state

%K_coupling = 5;
%class(K_input);
K_coupling = K_input;
%K_coupling = 5;
%O2_alpha = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate Neural Adjacency Matrix
A_neural = gen_adj(N_neurons, p, rho); %ER network
A_neural(eye(size(A_neural,1))==1) = 0; %Delete self-loops
A_neural = abs(A_neural);

%Simplified Kuramoto model version
A_neural = 0.5*(A_neural+A_neural');
A_neural(A_neural~=0) = 1;
A_neural = sparse(A_neural);

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

% Generate transition probabilities for Diffusion Layer
%Diffusion_Transition_Matrix = zeros(N_nodes,N_nodes);
%deg = sum(Diffusion_Lattice,2);
%if p_stay fixed for all nodes
%Diffusion_Transition_Matrix = p_move*Diffusion_Lattice./deg + p_stay*eye(N_nodes,N_nodes);
%if staying is just as likely as moving in any other direction
%Diffusion_Transition_Matrix = (Diffusion_Lattice + eye(N_nodes,N_nodes))./(deg+1);
        

%Initialize O2 content of each node in the flow layer, as well as the
%amount of O2 available for diffusion into the diffusion layer
Node_O2_Flow = zeros(N_nodes,tsteps+1);
Diffusion = zeros(size(Current_Sinks,1),tsteps+1);
Node_O2_Flow(Current_Source) = I_O2;

%Initialize the O2 content of each node in the Diffusion layer
%Node_O2_Diff = 0.8*O2_bath*ones(N_nodes,tsteps+1);
Node_O2_Diff = zeros(N_nodes,tsteps+1);

%Initialize data variables for the neural layer
Voltages = zeros(N_neurons,tsteps);
Phases = zeros(N_neurons,tsteps);
Phase_vel = zeros(N_neurons,tsteps);
Phases(:,t_start) = 2*pi*rand(N_neurons,1);
%Fire_mat = zeros([N_neurons,N_neurons,tsteps]);
Voltage_thresh = normrnd(20e-3,1e-3,[N_neurons,1]);
%Fired = zeros(N_neurons,tsteps);
Active = ones([N_neurons,tsteps]);
%O2 = 0.8*O2_bath*ones(N_neurons,tsteps);
O2 = zeros(N_neurons,tsteps);

% Define variable electrical properties for each neuron (assuming normal
% distribution)
Capacitances = normrnd(Mean_Capacitance,Var_Capacitance,[N_neurons,1]);
Resistances = normrnd(Mean_Resistance,Var_Resistance,[N_neurons,1]);
Natural_Freqs = zeros(N_neurons,1);
for i=1:N_neurons
    Natural_Freqs(i) = 1/(Capacitances(i)*Resistances(i)); %doesn't really make sense as a frequency, but it does define a time scale, so sort of close
end

% Decide which nodes in the diffusion layer each neuron will be connected
% to
Random_List = randperm(size(Diffusion_Lattice,1));
Adj_nd = zeros(N_neurons,size(Node_O2_Diff,1));
for i=1:N_neurons
    Adj_nd(i,Random_List(i)) = 1;
end
%Diff_to_Neural_Cxns = Random_List(1:N_neurons);
Adj_nd = sparse(Adj_nd);
Flow_Lattice = sparse(Flow_Lattice);
Diffusion_Lattice = sparse(Diffusion_Lattice);

%% Update loop
% NOTE: There's something a bit weird about time sequencing here. Since
% update_diff and update_neural both change the oxygen content in the
% diffusion layer, we have to decide which happens first. As it is, first
% the diffusion layer updates, then oxygen is pulled from the diffusion
% layer. I don't think the order should matter much, but it's worth noting
% in case I find later that it does matter.
for i=1:(tsteps-1)
    %{
    if(i < 50)
        [Node_O2_Flow(:,i+1),Diffusion(:,i+1)] = update_flow(Node_O2_Flow(:,i),Transition_matrix,Current_Source,Current_Sinks,I_O2);
        Node_O2_Diff(:,i+1) = update_diff(Node_O2_Diff(:,i),Diffusion_Lattice,Diffusion(:,i),Current_Sinks,dt,D_diff,D_flow_to_diff);
    end
    %}
    % Let the diffusion network get an O2 supply before turning on the
    % neural layer
    %if(i >= 50)
        %[Active(:,i), Fired(:,i), Voltages(:,i), O2(:,i),Node_O2_Diff(:,i+1)] = update_neural_if(Active(:,i-1), Fired(:,i-1), Voltages(:,i-1), O2(:,i-1), Resistances, Capacitances, A_neural, dt, K, 9e-5, Voltage_thresh,Node_O2_Diff(:,i+1),Diff_to_Neural_Cxns,D_diff_to_neural,O2_need);
        %[Phases(:,i+1), O2(:,i+1), Node_O2_Diff(:,i+1),Phase_vel(:,i+1)] = update_neural_kuramoto(Phases(:,i), O2(:,i), Natural_Freqs, A_neural, dt, K_coupling,Node_O2_Diff(:,i+1),Adj_nd,D_diff_to_neural,O2_avg,Beta,tanh_spread);
        [Phases(:,i+1), O2(:,i+1), Node_O2_Diff(:,i+1),Phase_vel(:,i+1)] = update_system(Phases(:,i), O2(:,i), Natural_Freqs, A_neural, dt, K_coupling,Node_O2_Diff(:,i),Adj_nd,D_diff_to_neural,O2_avg,Beta,tanh_spread,Diffusion_Lattice,Current_Sinks,D_diff,D_flow_to_diff,O2_bath);
    %end
end

%% Calc Numeric Output
%{
order_param = zeros(tsteps,1);
for t=1:tsteps
    for i=1:N_neurons
        order_param(t) = order_param(t) + exp(1j*Phases(i,t));
    end
end
order_param = abs(order_param)/N_neurons;
%}
sys_order_param = calc_n_step_order_param_vs_t(Phases,A_neural,20);
one_step_order_param = calc_n_step_order_param_vs_t(Phases,A_neural,1);
one_step_avg_order_param = mean(one_step_order_param(:,(end-t_stable):end),2);
sys_avg_order_param = mean(sys_order_param((end-t_stable):end));
sys_var_order_param = var(sys_order_param((end-t_stable):end));
Node_O2_time_avg = 1/t_stable*(sum(Node_O2_Diff(:,(end-t_stable):end),2));
avg_O2 = mean(Node_O2_time_avg);
var_O2 = var(Node_O2_time_avg);
avg_vel = mean(Phase_vel,1);

%% Plots
figure(1);
t = 1:tsteps;
yyaxis left
plot(t,avg_vel);
yyaxis right
plot(t,sys_order_param(1,:));

figure(2);
plot(one_step_avg_order_param);
refline(0,sys_avg_order_param);
ylim([0,1]);

figure(3);
plot(t,sys_order_param(1,:),t,one_step_order_param(1,:));

%{
K_coupling;
avg_order_param;
figure(2);
plot(Phases(1,1:end));
figure(1);
plot(order_param);
figure(3);
plot(Phase_vel(1,1:end));

%{
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
%}


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
Diffusion_Graph.Nodes.Size = Node_O2_Diff(:,size(Node_O2_Diff,2));
title('Diffusion O_2 Content vs. Time');
xlabel('Time (ms)');
ylabel('Node');
colorbar;

figure(3);
plot2 = plot(Diffusion_Graph,'XData',Nodes_X,'YData',Nodes_Y);
for i=1:N_nodes
    highlight(plot2,i,'MarkerSize',10*Diffusion_Graph.Nodes.Size(i)/max(Diffusion_Graph.Nodes.Size));
    for j=1:N_neurons
        if(i==Random_List(j))
            highlight(plot2,i,'NodeColor','g');
        end
    end
end
title('Diffusion Layer (Size = O_2 Content at end of Sim, Color = Cxn to Neuron)');
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
set(1,'Position',[0,1000,600,300]);
set(2,'Position',[300,127,300,300]);
set(3,'Position',[0,128,300,300]);
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

function [phi_dot_vec] = calc_phi_dots(Natural_Freqs,S,O2_neur_vec,O2_avg,Phases,A_neural,K_coupling) 
Sin_matrix = sin(Phases-Phases');
Interaction_vec = sum(A_neural.*Sin_matrix',2);
tanh_O2 = tanh(S*(O2_neur_vec - O2_avg));
phi_dot_vec = Natural_Freqs.*(tanh_O2+1) + K_coupling*Interaction_vec;
end

function [O2_neur_dot_vec] = calc_O2_neur_dots(O2_diff_vec,O2_neur_vec,Adj_nd,D_nd,phi_dot_vec,Beta)

O2_neur_dot_vec = D_nd*sum(Adj_nd.*(O2_diff_vec-O2_neur_vec')',2) - Beta*phi_dot_vec;

end

function [O2_diff_dot_vec] = calc_O2_diff_dots(O2_diff_vec,O2_bath,Current_Sinks,Adj_diff,Adj_nd,O2_neur_vec,D_nd,D_diff,D_fd)
O2_bath_vec = O2_diff_vec;
O2_bath_vec(Current_Sinks) = O2_bath;
O2_diff_dot_vec = D_fd*(O2_bath_vec - O2_diff_vec) - D_diff*sum(Adj_diff.*(O2_diff_vec - O2_diff_vec'),2) - D_nd*sum(Adj_nd.*(O2_diff_vec - O2_neur_vec')',1)';
end

function [Phase_next, O2_next, Diff_O2_next,Phase_vel] = update_system(Phase_current, O2_current, Natural_Freqs, A_neural, dt, K,Node_O2_diff,Adj_nd,D_nd,O2_avg,Beta,tanh_spread,Diffusion_Lattice,Current_Sinks,D_diff,D_transfer,O2_bath)
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
phi_dot_vec_0 = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current,O2_avg,Phase_current,A_neural,K);
k0 = dt*phi_dot_vec_0;
l0 = dt*calc_O2_diff_dots(Node_O2_diff,O2_bath,Current_Sinks,Diffusion_Lattice,Adj_nd,O2_current,D_nd,D_diff,D_transfer);
m0 = dt*calc_O2_neur_dots(Node_O2_diff,O2_current,Adj_nd,D_nd,phi_dot_vec_0,Beta);

phi_dot_vec_1 = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current+0.5*m0,O2_avg,Phase_current+0.5*k0,A_neural,K);
k1 = dt*phi_dot_vec_1;
l1 = dt*calc_O2_diff_dots(Node_O2_diff+0.5*l0,O2_bath,Current_Sinks,Diffusion_Lattice,Adj_nd,O2_current+0.5*m0,D_nd,D_diff,D_transfer);
m1 = dt*calc_O2_neur_dots(Node_O2_diff+0.5*l0,O2_current+0.5*m0,Adj_nd,D_nd,phi_dot_vec_1,Beta);

phi_dot_vec_2 = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current+0.5*m1,O2_avg,Phase_current+0.5*k1,A_neural,K);
k2 = dt*phi_dot_vec_2;
l2 = dt*calc_O2_diff_dots(Node_O2_diff+0.5*l1,O2_bath,Current_Sinks,Diffusion_Lattice,Adj_nd,O2_current+0.5*m1,D_nd,D_diff,D_transfer);
m2 = dt*calc_O2_neur_dots(Node_O2_diff+0.5*l1,O2_current+0.5*m1,Adj_nd,D_nd,phi_dot_vec_2,Beta);

phi_dot_vec_3 = calc_phi_dots(Natural_Freqs,tanh_spread,O2_current+m2,O2_avg,Phase_current+k2,A_neural,K);
k3 = dt*phi_dot_vec_3;
l3 = dt*calc_O2_diff_dots(Node_O2_diff+l2,O2_bath,Current_Sinks,Diffusion_Lattice,Adj_nd,O2_current+m2,D_nd,D_diff,D_transfer);
m3 = dt*calc_O2_neur_dots(Node_O2_diff+l2,O2_current+m2,Adj_nd,D_nd,phi_dot_vec_3,Beta);

Phase_next = wrapTo2Pi(Phase_current+1/6*(k0+2*k1+2*k2+k3));
O2_next = O2_current+1/6*(m0+2*m1+2*m2+m3);
Diff_O2_next = Node_O2_diff+1/6*(l0+2*l1+2*l2+l3);
Phase_vel = 1/(6*dt)*(phi_dot_vec_0+2*phi_dot_vec_1+2*phi_dot_vec_2+phi_dot_vec_3);
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
