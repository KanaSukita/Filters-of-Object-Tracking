function [ x_hat, P_hat, K ] = KalmanPG( y,sys)

% y is observation data. Keep in mind that in real systems y will be updated in real time
% sys is a struct containing all relevant matricies

% dimensions
[Nobs Ntime] =size(y);
Nx=length(sys.x0);
M=sys.M; % state eq matrix
H=sys.H; % observation eq matrix
R=sys.R; % observation noise
Q=sys.Q; % prediction noise

K              =zeros(Nx,Nx,Ntime);  
P_p            =zeros(Nx,Nx,Ntime);
P_hat          =zeros(Nx,Nx,Ntime);
x_hat          =zeros(Nx,Ntime);
x_p            =zeros(Nx,Ntime);

%initial states
x_hat(:,1)     =sys.x0;
P_hat(:,:,1)   =sys.P0;
    for iobs=2:Ntime
    % predict
        x_p(:,iobs)     = M * x_hat(:,iobs - 1);%your prediction of x
        P_p(:,:,iobs)   = M * P_hat(:,:,iobs - 1) * M' + Q;%your covariance of prediction
    % update
        Ptemp           = P_p(:,:,iobs);
        K(:,:,iobs)     = Ptemp * H' / (H * Ptemp * H' + R); %your Kalman gain
        P_hat(:,:,iobs) = Ptemp - K(:,:,iobs) * H * Ptemp;%your covariance update
        x_hat(:,iobs)   = x_p(:,iobs) + K(:,:,iobs) * (y(:,iobs) - H * x_p(:,iobs));%your new x prediction
    end
end