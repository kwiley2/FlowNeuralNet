% Simulate Flow + Diffusion dynamics

function [avg_order_param] = FlowDiffusionNeuralSim(K_input)

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Free Simulation Parameters (Other parameters, such as the underlying
% lattice structure and flow network details exist, but are altered in
% the piece of code which generates the flow network)

tsteps = 2^13; % Number of steps
dt = 2e-5; % Step Size

% Amount of O2 that enters flow layer at source node per time step
I_O2 = size(Current_Sinks,1)*100;

% Diffusion Coefficients (if continuous)
D_flow_to_diff = 50;
D_diff = 50;
D_diff_to_neural = 50;

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
O2_consumption = 1; % Amount of O2 necessary to transition back into active state

%K_coupling = 5;
%class(K_input);
K_coupling = K_input;
%K_coupling = 5;
O2_alpha = 0;

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
Node_O2_Diff = zeros(N_nodes,tsteps+1);

%Initialize data variables for the neural layer
Voltages = zeros(N_neurons,tsteps);
Phases = zeros(N_neurons,tsteps);
Fire_mat = zeros([N_neurons,N_neurons,tsteps]);
Voltage_thresh = normrnd(20e-3,1e-3,[N_neurons,1]);
Fired = zeros(N_neurons,tsteps);
Active = ones([N_neurons,tsteps]);
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
%NOTE: There's something a bit weird about time sequencing here. Since
%update_diff and update_neural both change the oxygen content in the
%diffusion layer, we have to decide which happens first. As it is, first
%the diffusion layer updates, then oxygen is pulled from the diffusion
%layer. I don't think the order should matter much, but it's worth noting
%in case I find later that it does matter.
for i=1:tsteps
    [Node_O2_Flow(:,i+1),Diffusion(:,i+1)] = update_flow(Node_O2_Flow(:,i),Transition_matrix,Current_Source,Current_Sinks,I_O2);
    Node_O2_Diff(:,i+1) = update_diff(Node_O2_Diff(:,i),Diffusion_Lattice,Diffusion(:,i),Current_Sinks,dt,D_diff,D_flow_to_diff);
    %Let the diffusion network get an O2 supply before turning on the
    %neural layer
    if(i > 200)
        %[Active(:,i), Fired(:,i), Voltages(:,i), O2(:,i),Node_O2_Diff(:,i+1)] = update_neural_if(Active(:,i-1), Fired(:,i-1), Voltages(:,i-1), O2(:,i-1), Resistances, Capacitances, A_neural, dt, K, 9e-5, Voltage_thresh,Node_O2_Diff(:,i+1),Diff_to_Neural_Cxns,D_diff_to_neural,O2_need);
        [Phases(:,i+1), O2(:,i+1), Node_O2_Diff(:,i+1)] = update_neural_kuramoto(Phases(:,i), O2(:,i), Natural_Freqs, A_neural, dt, K_coupling,Node_O2_Diff(:,i+1),Adj_nd,D_diff_to_neural,O2_consumption,O2_alpha);
    end
end

%% Plotting

order_param = zeros(tsteps,1);
for t=200:tsteps
    for i=1:N_neurons
        order_param(t) = order_param(t) + exp(1j*Phases(i,t));
    end
end
order_param = abs(order_param)/N_neurons;
avg_order_param = mean(order_param((end-6000):end));
K_coupling;
avg_order_param;
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

%Node_O2_next = zeros(size(Node_O2_current,1),1);
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

%Node_O2_next = zeros(size(Node_O2_current,1),1);
Node_O2_next = Node_O2_current;
Node_O2_difference = Node_O2_current-Node_O2_current';
%Node_O2_transfer_vec = diag(Diffusion_Lattice*Node_O2_difference);
Node_O2_transfer_vec = sum(Diffusion_Lattice'.*Node_O2_difference,1)';
Node_O2_next = Node_O2_current + dt*D_diff*Node_O2_transfer_vec;

Diffusable_vec = zeros(size(Node_O2_current,1),1);
Diffusable_vec(Current_Sinks) = Diffusable;

diffuse = Node_O2_current < Diffusable_vec;
Node_O2_next = Node_O2_next + dt*D_transfer*(Diffusable_vec - Node_O2_current).*diffuse;

%Continuous Diffusion
%{
for i=1:size(Node_O2_current,1)
    %{
    for j=1:i
        if(Diffusion_Lattice(i,j) ~= 0)
            Node_O2_next(i) = Node_O2_next(i) - D_diff*(Node_O2_current(i)-Node_O2_current(j))*dt;
            Node_O2_next(j) = Node_O2_next(j) + D_diff*(Node_O2_current(i)-Node_O2_current(j))*dt;
        end
    end
     %}
    
    %if updating a sink node and diffusion layer has less O2 than the flow
    %layer, then diffusion occurs from the flow layer to the diffusion
    %% 
    %% 
    %layer
    if(sum(Current_Sinks==i)>0)
        if(Node_O2_current(i) < Diffusable(find(Current_Sinks==i)))
            N_O2_transfer = D_transfer*(Diffusable(find(Current_Sinks==i))-Node_O2_current(i))*dt;
            Node_O2_next(i) = Node_O2_next(i) + N_O2_transfer;
        end
    end
end
%}

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

function [Phase_next, O2_next, Diff_O2_next] = update_neural_kuramoto(Phase_current, O2_current, Natural_Freqs, A_neural, dt, K,Node_O2_diff,Adj_nd,D_nd,O2_consumption,O2_alpha)
% This just preallocates size. The values change in a way that doesn't rely
% on the current state, so the specific values here don't matter
Phase_next = Phase_current;
O2_next = O2_current;
%Active_next = Active_current;
%Fired_next = zeros(size(Fired_current,1),1);
Diff_O2_next = Node_O2_diff;
O2_rest = 0;
alpha = O2_alpha;

%Fire_mat = eye(size(Fired_current,1)).*Fired_current;
N_neurons = size(Phase_current,1);
%Sin_matrix = zeros(N_neurons,N_neurons);
%Phase_diff_mat = Phase_current-Phase_current';
Sin_matrix = sin(Phase_current-Phase_current');
%Sin_matrix = bsxfun(@(x,y) sin(x-y),Phase_current,Phase_current');
%{
for i=1:N_neurons
    for j=1:i
        Sin_matrix(i,j) = sin(Phase_current(i) - Phase_current(j));
    end
end
Sin_matrix = Sin_matrix - Sin_matrix';
%}
%Interaction_vec = diag(A_neural*Sin_matrix); %diagonal entries are the interaction terms in the Kuramoto model
Interaction_vec = sum(A_neural'.*Sin_matrix,1)';

%O2_diff_nd = zeros(size(Node_O2_diff,1),N_neurons);
O2_diff_nd = Node_O2_diff-O2_current';
%{
for j=1:N_neurons
    for i=1:size(Node_O2_diff,1)
        O2_diff_nd(i,j) = Node_O2_diff(i) - O2_current(j);
    end
end
%}
%Interlayer_O2_vec = diag(Adj_nd*O2_diff_nd); %diagonal entries are the diffusion terms between layers
%Adj_nd_temp = sparse(Adj_nd);
%[~,Adj_screen] = ind2sub(size(Adj_nd),find(Adj_nd))

%Interlayer_O2_vec = O2_diff_nd(:,Adj_nd_screen);

%Interlayer_O2_vec = Interlayer_O2_vec';
Interlayer_O2_vec = sum(Adj_nd'.*O2_diff_nd,1)';
%Interlayer_O2_vec = diag(Interlayer_O2_matrix); %put just the interesting bits in a vector

%{
for j=1:N_neurons
    Fired = 0;
    %k1 = dt*dphase_dt(Natural_Freqs(j),K,Interaction_matrix(j,j),alpha,O2_rest,O2_current(j));
    %implementing a Runge-Kutta method would be nice, but really difficult
    %because of the complexity of the system of equations and because of
    %the Dirac delta O2 change when the phase crosses 2pi
    %Phase_next(j) = Phase_current(j) + dt*dphase_dt(Natural_Freqs(j),K,Interaction_vec(j),alpha,O2_rest,O2_current(j));
    Fired = 0;
    Phase_next(j) = Phase_current(j) + dt*(Natural_Freqs(j) + K*Interaction_vec(j))*exp(-alpha*(O2_rest - O2_current(j)));
    while(Phase_next(j) >= 2*pi)
        Phase_next(j) = Phase_next(j) - 2*pi;
        Fired = 1;
    end
    O2_next(j) = O2_current(j) + dt*(D_nd*Interlayer_O2_vec(j)) - O2_consumption*Fired;
    diffusion_node = find(Adj_nd(j,:)==1);
    Diff_O2_next(diffusion_node) = Node_O2_diff(diffusion_node) - dt*(D_nd*Interlayer_O2_vec(j));
end
%}
%trying to implement the above for loop using matrix methods instead

%exp_O2 = arrayfun(@(x) exp(-alpha*(O2_rest-x)),O2_current);
exp_O2 = exp(-alpha*O2_rest)*exp(alpha*O2_current);
phi_dot = (Natural_Freqs + K*Interaction_vec).*exp_O2;
Phase_next = Phase_current + dt*phi_dot;
Fired = Phase_next>(2*pi);
for j=1:N_neurons
    while(Phase_next(j)>2*pi)
        Phase_next(j) = Phase_next(j)-2*pi;
    end
end
%Phase_next = arrayfun(@(x) mod(x,2*pi),Phase_next);
O2_next = O2_current + dt*D_nd*Interlayer_O2_vec - O2_consumption*Fired;
Diff_O2_next = Node_O2_diff - dt*D_nd*(Adj_nd'*Interlayer_O2_vec);



end

function [output] = dphase_dt(natural_freq, K, interaction,alpha,O2_rest,O2_current) 
output = (natural_freq+K*interaction)*exp(-alpha*(O2_rest-O2_current));
end

