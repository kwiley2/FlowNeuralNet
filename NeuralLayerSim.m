% Leaky Integrate-and-Fire Neural Network

%Intuitive Idea:
%Each neuron i has an assigned voltage at each time t, V_i(t). 
%At each time step, the neuron recieves some amount of current
%which is distributed simply as noise of some kind. E.g. each neuron
%recieves current which is probabilistically distributed according to white
%noise. For the sake of argument, suppose that no neuron had fired at the
%previous time step. Then the voltage at neuron i changes depending on this
%incoming current, as well as on the "leaky" current which disappears with
%time constant C_i*R_i. The conversion from current to change in voltage is
%simply given by the capacitance C_i. If the voltage then crosses a given
%threshold T_i, the neurons voltage at the next time step will be set to
%zero and it will fire an action potential. Now, if a neuron j fired at the
%previous time step and connects to neuron i, neuron i will recieve an
%additional current proportional to the strength of the edge pointing from
%j to i. That covers the neuron-neuron dynamics.
%Now we will also suppose that each neuron, once it fires, requires some
%amount of O2 before it can fire again, in order to reinstate the ionic
%gradient using energy-driven ion channels. We will call this quantity
%O_t^i, which stands for threshold oxygen at neuron i. Once a neuron fires,
%it will begin to pull oxygen from the connected nodes in the diffusion
%layer proportional to the amount of oxygen in those nodes (i.e.
%diffusively, but we assume the neuron always has no partial pressure of
%O2). Until it pulls enough oxygen to cross threshold, it will accumulate
%voltage as normal, but be unable to fire. At a later date, we may
%additionally impose that if it doesn't cross the threshold after some
%amount of time, it dies and remains dormant for the remainder of the
%simulation.

%We can write this more briefly as the following set of equations:

%V_i(t+1) = V_i(t) + [(I_noise(t)/C)_i - (V(t)/(RC))_i]*dt + (Sum_j(I_fire^(j->i)(t)))_i
%where I_fire^(j->i) = k(FA)_ij, F is the identity matrix but with the 1 at
%column i set to zero if neuron i didn't fire, A is the weighted adjacency
%matrix, and k is a conversion factor between weight and current
%If V_i(t+1) > T_i && D_i = 0, V_i(t+1) = 0, F_ii(t+1) = 1, D_i(t+1) = 1 (D_i is a
%variable indicating that neuron i is now dormant), and O_i(t+1) = -O_t^i
%If O_i(t) < 0, O_i(t+1) = O_i(t) + D_inter*O_diff_connected*dt
%If O_i(t) < 0 && O_i(t+1) > 0, D_i(t+1) = 0

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Network = load('SampleFlowNetwork');

% Step sizes of network lattice (useful for plotting)
delta_x = Network.FlowNetwork.LatticeDeltaX;
delta_y = Network.FlowNetwork.LatticeDeltaY;

% Load in Diffusion and Flow networks, as well as which nodes were used
% as sources and sinks when the flow network was constructed
Diffusion_Lattice = Network.FlowNetwork.LatticeAdj;
Flow_Lattice = Network.FlowNetwork.ConductanceNetwork;
Current_Source = Network.FlowNetwork.CurrentSource;
Current_Sinks = Network.FlowNetwork.CurrentSinks';
Nodes_X = Network.FlowNetwork.X_positions;
Nodes_Y = Network.FlowNetwork.Y_positions;

N_nodes = size(Diffusion_Lattice,1);

n_steps = 2048;
dt = 1e-3; %ms
K = 1.7e-4; %Conversion between weights and currents
%K = 0;

N_neurons = 100;% number of neurons
p = 0.9*1/sqrt(N_neurons); %probability of a given edge existing
rho = 1.4;

A_neural = gen_adj(N_neurons, p, rho); %ER network
A_neural(eye(size(A_neural,1))==1) = 0; %Delete self-loops
A_neural = abs(A_neural);

Voltages = zeros(N_neurons,n_steps);
Fire_mat = zeros([N_neurons,N_neurons,n_steps]);
Voltage_thresh = normrnd(20e-3,1e-3,[N_neurons,1]);
Fired = zeros(N_neurons,n_steps);
Active = ones([N_neurons,n_steps]);
O2 = zeros(N_neurons,n_steps);

% Define variable electrical properties for each neuron
Capacitances = normrnd(20e-6,3e-6,[N_neurons,1]); %capacitances ~20uF
Resistances = normrnd(500,100,[N_neurons,1]); %resistances ~500 Ohms

for i=2:n_steps
    [Active(:,i), Fired(:,i), Voltages(:,i), O2(:,i)] = update_neural(Active(:,i-1), Fired(:,i-1), Voltages(:,i-1), O2(:,i-1), Resistances, Capacitances, A_neural, dt, K, 9e-5, Voltage_thresh);
end

Net_Activity = sum(Fired,1);
y_net = fft(Net_Activity);
f_net = (0:length(y_net)-1)*50/length(y_net);

y_1 = fft(Fired(1,:));
f_1 = (0:length(y_1)-1)*50/length(y_1);

figure(1);
imagesc(Fired);
figure(2);
plot(Net_Activity);
xlim([0 n_steps]);
figure(3);
plot(f_net(2:floor(end/2)),abs(y_net(2:floor(end/2))));
xlim([0 5]);

set(1,'Position',[600,1000,800,300]);
set(2,'Position',[600,300,800,150]);
set(3,'Position',[600,90,800,150]);

function [Active_next, Fired_next, Voltages_next, O2_next] = update_neural(Active_current, Fired_current, Voltages_current, O2_current, Resistances, Capacitances, A_neural, dt, K, Noise_variance, Voltage_thresh) 
% This just preallocates size. The values change in a way that doesn't rely
% on the current state, so the specific values here don't matter
Voltages_next = Voltages_current;
O2_next = O2_current;
Active_next = Active_current;
Fired_next = zeros(size(Fired_current,1),1);

Fire_mat = eye(size(Fired_current,1)).*Fired_current;
I_fire = K*Fire_mat*A_neural;
N_neurons = size(Voltages_current,1);

for j=1:N_neurons
    Voltages_next(j) = Voltages_current(j) + (normrnd(0,Noise_variance)/Capacitances(j)-Voltages_current(j)/(Capacitances(j)*Resistances(j)))*dt + dt*sum(I_fire(:,j))/Capacitances(j);
    if(O2_current(j) < 0)
        O2_next(j) = O2_current(j)+1;
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
        O2_next(j) = -5;
        Active_next(j) = 0;
    end
end
end