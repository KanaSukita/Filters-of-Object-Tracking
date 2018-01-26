%% Parameters

F_update = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];

Npop_particles = 1000;
%"Xstd_rgb" means standard deviation of observation noise, 
%which means noise you get when you observe the state of something.
%"Xstd_pos" and "Xstd_vec" mean standard deviation of system noise, 
%which describes how far actual movement of target object differs from the 
%ideal model (in this case, linear uniform motion).
Xstd_rgb = 50;
Xstd_pos = 25;
Xstd_vec = 15;

Xrgb_trgt = [0; 255; 0];

%% Loading Movie

vr = VideoReader('Green.mp4');

Npix_resolution = [vr.Height vr.Width];
%Nfrm_movie = floor(vr.Duration * vr.FrameRate);

%% Object Tracking by Particle Filter

X = create_particles(Npix_resolution, Npop_particles);

% Record detected coordinates
Particle_Coord = [];

% Save to video
%vv = VideoWriter('Particle Results_Green.mp4','MPEG-4');
%open(vv);
%for k = 1:Nfrm_movie
while hasFrame(vr)    
    % Getting Image
    %Y_k = read(vr, k);
    Y_k = readFrame(vr);
    
    % Forecasting
    X = update_particles(F_update, Xstd_pos, Xstd_vec, X);
    
    % Calculating Log Likelihood
    L = calc_log_likelihood(Xstd_rgb, Xrgb_trgt, X(1:2, :), Y_k);
    
    % Resampling
    X = resample_particles(X, L);

    % Showing Image
    % show_particles(X, Y_k); 
    % show_center(X, Y_k);
    figure(1)
    imshow(Y_k)
    title('Particle Tracking Result')

    hold on
    x_center = mean(X(2,:));
    y_center = mean(X(1,:));
    plot(x_center, y_center, '+', 'MarkerEdgeColor', 'y', 'MarkerSize', 50)
    hold off
    
    % Record Coordinates
    Particle_Coord = [Particle_Coord, [x_center; y_center]];

    drawnow
    %show_state_estimated(X, Y_k);

    % Record frame into video
    %F = getframe;
    %writeVideo(vv, F);

end
%close(vv);
%% Save the tracked Coordinates
path = 'C:\Users\SakuyasPad\Documents\MATLAB\ECE251B\Project\Evaluation\';
save([path, 'Particle_Coord'] ,'Particle_Coord');
