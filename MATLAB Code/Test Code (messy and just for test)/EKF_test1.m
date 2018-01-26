vr = vision.VideoFileReader('mouse_2_Loss.mp4');
load('LossCount.mat')

%% Set Parameters
param.motionModel           = 'ConstantAcceleration';
param.initialLocation       = 'Same as first detection';
param.initialEstimateError  = 1E5 * ones(1, 3);
param.motionNoise           = [25, 10, 1];
param.measurementNoise      = 25;
param.segmentationThreshold = 0.05;

% Change Model
%{
param.motionModel           = 'ConstantVelocity';
param.initialEstimateError  = param.initialEstimateError(1:2);
param.motionNoise           = param.motionNoise(1:2);
%}

% Change Model 2
%{
param.initialLocation = [0, 0];  % location that's not based on an actual detection 
param.initialEstimateError = 100*ones(1,3); % use relatively small values
%}

% Change Model 3
%{
param.segmentationThreshold = 0.0005; % smaller value resulting in noisy detections
param.measurementNoise      = 12500;  % increase the value to compensate 
                                      % for the increase in measurement noise
%}

%% Set Foreground Detector
foregroundDetector = vision.ForegroundDetector(...
    'NumTrainingFrames', 10, 'InitialVariance', param.segmentationThreshold);
blobAnalyzer = vision.BlobAnalysis('AreaOutputPort', false, ...
    'MinimumBlobArea', 70, 'CentroidOutputPort', true);

isTrackInitialized = false;
KF_Coord = [];

while ~isDone(vr)
    Frame = step(vr);
    grayImage = rgb2gray(Frame);
    foregroundMask = step(foregroundDetector, grayImage);
    detection = step(blobAnalyzer, foregroundMask);
    figure(1);
    imshow(Frame);
    %combinedImage = max(repmat(foregroundMask, [1,1,3]), Frame);
    %imshow(combinedImage);
    hold on;
    if ~isempty(detection)
        detection = detection(1, :);
        plot(detection(1), detection(2), '+', 'MarkerEdgeColor', 'y', 'MarkerSize', 10);
        isObjectDetected = true;
    else
        isObjectDetected = false;
    end
    
    if ~isTrackInitialized
        if isObjectDetected
            % Initialize a track by creating a Kalman filter when the ball is
            % detected for the first time.
            if strcmp(param.initialLocation, 'Same as first detection')
                initialLocation = detection;
            else
                initialLocation = param.initialLocation;
            end
            x=[initialLocation(1);0;0;initialLocation(2);0;0];
            f = @(x)[x(1)+x(2)*x(7)+x(3)/(2*x(7)^2);x(2)+x(3)*x(7);x(3);x(4)+x(5)*x(7)+x(6)/(2*x(7)^2);x(5)+x(6);x(6);x(7)];
            h=@(x)[x(1);x(4)];                              
            P=diag([param.initialEstimateError param.initialEstimateError 100000]);
            Q=diag([param.motionNoise param.motionNoise 1]);
            R=diag([param.measurementNoise param.measurementNoise]);
            
            isTrackInitialized = true;
            trackedLocation = [x(1); x(4)];
            label = 'Initial';
            framenum=1;
            plot(trackedLocation(1), trackedLocation(2), 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 10);
        else
            trackedLocation = [];
            label = '';
        end
        
    else
        % Use the Kalman filter to track the ball.
        if isObjectDetected % The ball was detected.
            % Reduce the measurement noise by calling predict followed by
            % correct.
            if(framenum==1)
                framenum=framenum+1;
                t=LossCount(framenum)+1;
                x(2)=(detection(1)-x(1))/t;
                x(5)=(detection(2)-x(4))/t;
                x(1)=detection(1);
                x(4)=detection(2);
            else
                if(framenum==2)
                    framenum=framenum+1;
                    t=LossCount(framenum)+1;
                    x(3)=2*(detection(1)-x(1)-t*x(2))/(t^2);
                    x(6)=2*(detection(2)-x(4)-t*x(5))/(t^2);
                    x(2)=x(2)+t*x(3);
                    x(5)=x(5)+t*x(6);
                    x(1)=detection(1);
                    x(4)=detection(2);
                else
                    framenum=framenum+1;
                    t=LossCount(framenum)+1;
                    z=[detection(1);detection(2)];
                    x(7)=t;
                    [x, P] = ekf(f,x,P,h,z,Q,R);
                end
            end
            x=real(x);
            trackedLocation = [x(1);x(4)];
            label = 'Corrected';
            plot(trackedLocation(1), trackedLocation(2), 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 10);
        else % The ball was missing.
            % Predict the ball's location.
            t=1;
            x=f(x);
            trackedLocation = [x(1);x(4)];
            label = 'Predicted';
            plot(trackedLocation(1), trackedLocation(2), 'o', 'MarkerEdgeColor', 'b', 'MarkerSize', 10);
        end
    end
    hold off;
    drawnow
    if ~isempty(trackedLocation)
        KF_Coord = [KF_Coord,[trackedLocation(1);trackedLocation(2)]];
    end
end

path = 'C:\Users\SakuyasPad\Documents\MATLAB\ECE251B\Project\Evaluation\';
save([path, 'KF_Coord'] ,'KF_Coord');