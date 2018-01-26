function X = update_particles(F_update, Xstd_pos, Xstd_vec, X)
%"Xstd_pos" and "Xstd_vec" mean standard deviation of system noise, 
%which describes how far actual movement of target object differs from the 
%ideal model (in this case, linear uniform motion).
N = size(X, 2);

X = F_update * X;

X(1:2,:) = X(1:2,:) + Xstd_pos * randn(2, N);
X(3:4,:) = X(3:4,:) + Xstd_vec * randn(2, N);
end