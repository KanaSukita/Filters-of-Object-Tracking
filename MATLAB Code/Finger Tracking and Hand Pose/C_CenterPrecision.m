%% Load GroundTruth
load('GroundTruth.mat');

%% Input Selection
% load('Particle_Coord.mat');
% load('KF_Coord.mat');
 load('EKF_Coord.mat');
% load('UKF_Coord.mat');
% result = Particle_Coord;
% result = KF_Coord;
 result = EKF_Coord;
% result = UKF_Coord;

%%
GroundTruth = GroundTruth(:,:);
result = result(:,:);
%%
Diff = result - GroundTruth;
Norm2Error = sum(Diff.^2).^0.5;
Norm2Error = sum(Norm2Error)/size(Norm2Error,2)

ShowTrajectory(GroundTruth, result);