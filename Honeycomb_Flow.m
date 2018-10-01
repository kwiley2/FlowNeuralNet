% Construct Flow Network Using Dissipation Minimization with Modular Flow
% TO DO: * Create a terminating condition, rather than fixed update number
%        * Find a way to sample only a subset of flow states so that you can
%             increase alpha and N_sinks and still get results

gamma = 0.4;
length=1;
delta_x = length/2;
delta_y = sqrt(3)*length/2;
N = 100; %Number of delta_x's
M = 50; %Number of delta_y's
fluctuating = 1;

% generate a hexagonal lattice which is superimposed on a rectangular
% lattice
rect_lattice = zeros(N,M);
rect_lattice(1,1) = 1;
rect_lattice(3,1) = 1;
rect_lattice(4,2) = 1;
rect_lattice(6,2) = 1;
for i=1:N
    for j=1:M
        if(j >= 3 && rect_lattice(i,j-2) == 1)
            rect_lattice(i,j) = 1;
        end
        if(i >= 7 && rect_lattice(i-2,j) == 1 && rect_lattice(i-6,j) == 1)
            rect_lattice(i,j) = 1;
        end
        if(i >= 7 && rect_lattice(i-4,j) == 1 && rect_lattice(i-6,j) == 1)
            rect_lattice(i,j) = 1;
        end
    end
end

% create a labeling of the nodes in the lattice (bottom to top, then left 
% to right)
Labels = zeros(N,M);
current_label = 1;
for i=1:N
    for j=1:M
        if(rect_lattice(i,j) == 1)
            Labels(i,j) = current_label;
            current_label = current_label + 1;
        end
    end
end

N_nodes = max(max(Labels));

Adj_mat = zeros(N_nodes);

% super wasteful way of generating adjacency matrix
for i=1:N
    for j=1:M
        for l=1:N
            for m=1:M
                if(rect_lattice(i,j) == 1 ...
                        && rect_lattice(l,m) == 1 ...
                        && floor(node_distance(i,j,l,m,delta_x,delta_y)+0.5)==1)
                    Node_1 = Labels(i,j);
                    Node_2 = Labels(l,m);
                    Adj_mat(Node_1,Node_2) = 1;
                    Adj_mat(Node_2,Node_1) = 1;
                end
            end
        end
    end
end

% give edges random weights
Weighted_Adj_mat = Adj_mat;
for i=1:N_nodes
    for j=1:i
        if(Adj_mat(i,j) == 1)
            weight = rand;
            Weighted_Adj_mat(i,j) = weight;
            Weighted_Adj_mat(j,i) = weight;
        end
    end
end

% Establish source and sink currents
N_sinks = floor(N_nodes/5); %fixed number
if(fluctuating)
    alpha = 1; %Number of "current quanta" to distribute over sinks
    %each row is a unique distribution of alpha quanta over N_sinks bins
    Distributions = find_distributions(alpha,N_sinks);
    N_states = size(Distributions,1);
else
    N_states = 1;
end
Current_Sources = zeros(N_nodes,N_states);
Current_Sources(1) = 1; %input current


% Decide how many current sinks there should be
%N_sinks = floor(rand*0.5*(N_nodes-1)); %random number
%randomly distribute those sinks over nodes 2:N_nodes
rand_nodes = 2:N_nodes;
rand_nodes = rand_nodes(randperm(N_nodes-1));
%%%%%%%%%%%%Current sinks are steady-state%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(~fluctuating)
    Current_Sources(rand_nodes(1:N_sinks)) = -1/N_sinks;
%%%%%%%%%%%%Current sinks are fluctuating%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    %Take transpose and divide by alpha to get columns and normalize
    Distributions = Distributions'/alpha;
    %Assign all possible distributions to our N_sinks randomly chosen sinks
    Current_Sources(rand_nodes(1:N_sinks),1:size(Distributions,2))=-Distributions;
    Current_Sources(1,:) = 1;
end    
Currents = zeros(N_nodes,N_nodes,N_states);

%Core Conductance update loop
% To Do: Create a termination condition, rather than a fixed # of loops
N_loops=100;
Subset_size = N_states;

%%%%%%%%%%%%Current sinks are steady-state%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(~fluctuating)
    for i=1:N_loops
        G = -Weighted_Adj_mat + eye(N_nodes).*sum(Weighted_Adj_mat,2);
        Voltages = pinv(G)*Current_Sources;
        for i=1:N_nodes
            for j=1:N_nodes
                Currents(i,j) = Weighted_Adj_mat(i,j)*(Voltages(i)-Voltages(j));
            end
        end
        denom = sum(sum((Currents.^2).^(gamma/(1+gamma))))^(1/gamma);
        Weighted_Adj_mat = (Currents.^2).^(1/(1+gamma))/denom;
    end
%%%%%%%%%%%%Current sinks are fluctuating%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    dA_old = 10000;
    dA_new = 10;
    steps = 0;
    while(abs((dA_old-dA_new)/dA_old)>0.001 && steps<1000)
    %for i=1:N_loops
        %rand_states = randperm(N_states);
        rand_states = 1:N_states;
        G = -Weighted_Adj_mat + eye(N_nodes).*sum(Weighted_Adj_mat,2);
        Voltages = pinv(G)*Current_Sources;
        for i=1:N_nodes
            for j=1:N_nodes
                %for k=1:size(Voltages,2)
                for k=1:Subset_size
                    Currents(i,j,rand_states(k)) = Weighted_Adj_mat(i,j)*(Voltages(i,rand_states(k))-Voltages(j,rand_states(k)));
                end
            end
        end
        %denom = 0;
        %temp_denom = sum(sum((Currents.^2).^(gamma/(1+gamma))));
        I2 = Currents.^2;
        I2_avg = zeros(N_nodes,N_nodes);
        for k=1:Subset_size
            I2_avg = I2_avg + I2(:,:,rand_states(k));
            
            %denom = denom + temp_denom(rand_states(l));
            %temp_adj_mat = (Currents.^2).^(1/(1+gamma));
            %Weighted_Adj_mat = Weighted_Adj_mat + temp_adj_mat(:,:,rand_states(l));
        end
        I2_avg = I2_avg/Subset_size;
        denom = (0.5*sum(sum(I2_avg.^(gamma/(1+gamma)))))^(1/gamma);
        Weighted_Adj_mat_old = Weighted_Adj_mat;
        Weighted_Adj_mat = I2_avg.^(1/(1+gamma))/denom;
        dA_old = dA_new;
        dA_new = sum(sum((Weighted_Adj_mat-Weighted_Adj_mat_old).^2));
        if(dA_old == 10)
            dA_old = 2*dA_new;
        end
        steps = steps + 1;
        %Weighted_Adj_mat = Weighted_Adj_mat/denom;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plotting
Network_Graph = graph(Adj_mat);
LWidths_Conductance = zeros(size(Network_Graph.Edges.Weight,1),1);
LWidths_Current = zeros(size(Network_Graph.Edges.Weight,1),1);
k = 1;
% Loop thru all possible edges by first going through 1-all, then 2-all but
% 1, then 3-all but 1&2, etc.
% For each, if an edge exists, get its weight and assign it to LWidths
% If the weight of an edge is very small (i.e. it atrophied), delete it
for i=1:N_nodes
    for j=i:N_nodes
        if(Adj_mat(i,j)==1)
            %if(Weighted_Adj_mat(i,j)>0.001*max(max(Weighted_Adj_mat)))
            if(Weighted_Adj_mat(i,j)>0)
                LWidths_Conductance(k) = Weighted_Adj_mat(i,j);
                LWidths_Current(k) = Currents(i,j);
                k=k+1;
            else
                Network_Graph = rmedge(Network_Graph,i,j);
                LWidths_Conductance(k) = [];
                LWidths_Current(k) = [];
            end
        end
    end
end
%Normalize for nice looking edge weights
LWidths_Conductance = 20*LWidths_Conductance/max(LWidths_Conductance);
LWidths_Current = 20*LWidths_Current/max(LWidths_Current);
%We only care about magnitude of current, not direction
LWidths_Current = abs(LWidths_Current);

%Find x-y positions of each node (easy b/c I created them from a physical
%lattice in the first place)
Nodes_X = zeros(size(Network_Graph.Nodes,1),1);
Nodes_Y = zeros(size(Network_Graph.Nodes,1),1);
for i=1:size(Nodes_X,1)
    [Nodes_X(i),Nodes_Y(i)] = ind2sub(size(Labels),find(Labels==i));
end
Nodes_X = Nodes_X*delta_x;
Nodes_Y = Nodes_Y*delta_y;

%Plot nodes with preassigned X and Y positions, edge weights adjusted by
%flow rate through that edge, nodes colored green if source, red if sink,
%and blue if neither
figure(1);
h2 = plot(Network_Graph,'XData',Nodes_X,'YData',Nodes_Y,'LineWidth',(LWidths_Conductance+0.0001));
highlight(h2,[1],'NodeColor','g');
highlight(h2,rand_nodes(1:N_sinks),'NodeColor','r');

FlowNetwork.LatticeAdj = Adj_mat;
FlowNetwork.X_positions = Nodes_X;
FlowNetwork.Y_positions = Nodes_Y;
FlowNetwork.LatticeDeltaX = delta_x;
FlowNetwork.LatticeDeltaY = delta_y;
FlowNetwork.CurrentSource = 1;
FlowNetwork.CurrentSinks = rand_nodes(1:N_sinks);
FlowNetwork.ConductanceNetwork = Weighted_Adj_mat;
FlowNetwork.NetworkPlot = h2;


function dist = node_distance(i1,j1,i2,j2,dx,dy)
x_dist = (i2-i1)*dx;
y_dist = (j2-j1)*dy;
dist = sqrt(x_dist^2+y_dist^2);
end