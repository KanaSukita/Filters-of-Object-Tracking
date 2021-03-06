vr = vision.VideoFileReader('RoundBall.mp4');

%% Set Parameters
param.motionModel           = 'ConstantAcceleration';
param.initialLocation       = 'Same as first detection';
param.initialEstimateError  = 1E5 * ones(1, 3);
param.motionNoise           = [25, 10, 1];
param.measurementNoise      = 2500;
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
UKF_Coord = [];

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
            f=@(x)[300+100*sin(0.2*(x(3)+1));200+100*cos(0.2*(x(3)+1));x(3)+1];  
            h=@(x)[x(1);x(2)];                              
            P=diag(param.initialEstimateError);
            Q=diag(param.motionNoise);
            R=diag(param.measurementNoise);
            
            isTrackInitialized = true;
            trackedLocation = [x(1); x(2)];
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
            z=[detection(1);detection(2)];
            [x, P] = ukf(f,x,P,h,z,Q,R);
            x=real(x);
            trackedLocation = [x(1);x(2)];
            label = 'Corrected';
            plot(trackedLocation(1), trackedLocation(2), 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 10);
        else % The ball was missing.
            % Predict the ball's location.
            x=f(x);
            trackedLocation = [x(1);x(2)];
            label = 'Predicted';
            plot(trackedLocation(1), trackedLocation(2), 'o', 'MarkerEdgeColor', 'b', 'MarkerSize', 10);
        end
    end
    hold off;
    drawnow
    if ~isempty(trackedLocation)
        UKF_Coord = [UKF_Coord,[trackedLocation(1);trackedLocation(2)]];
    end
end

path = 'C:\Users\SakuyasPad\Documents\MATLAB\ECE251B\Project\Evaluation\';
save([path, 'UKF_Coord'] ,'UKF_Coord');