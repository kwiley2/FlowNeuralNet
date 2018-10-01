% Distribute quanta probabilistically

function [dist_vec,diffusable] = dist_quanta(N_quanta,p_vec)
thresholds = zeros(size(p_vec,1),1);
thresholds(1) = p_vec(1);
for i=2:size(p_vec)
    thresholds(i) = thresholds(i-1) + p_vec(i);
end
rand_vec = rand(N_quanta,1);
dist_vec = zeros(size(thresholds,1),1);
dist_vec(1) = sum(rand_vec<thresholds(1));
for i=2:size(thresholds,1)
    dist_vec(i) = sum(rand_vec<thresholds(i)) - sum(rand_vec<thresholds(i-1));
end
%return also how much "disappears" into the capillaries and veins
diffusable = N_quanta - sum(dist_vec);