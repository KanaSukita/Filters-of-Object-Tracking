vr = vision.VideoFileReader('circle_acc.mp4');
%vw = VideoWriter('KF_circle.mp4','MPEG-4');
%open(vw);

%% Set Parameters
param.motionModel           = 'ConstantAcceleration';
param.initialLocation       = 'Same as first detection';
param.initialEstimateError  = 1E3 * ones(1, 3);
param.motionNoise           = [250, 250, 1];
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
    'NumTrainingFrames', 20, 'InitialVariance', param.segmentationThreshold);
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
            kalmanFilter = configureKalmanFilter(param.motionModel, ...
                initialLocation, param.initialEstimateError, ...
                param.motionNoise, param.measurementNoise);
            
            isTrackInitialized = true;
            trackedLocation = correct(kalmanFilter, detection);
            label = 'Initial';
            plot(trackedLocation(1), trackedLocation(2), 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 10);
        else
            trackedLocation = [-1,-1];
            label = '';
        end
        
    else
        % Use the Kalman filter to track the ball.
        if isObjectDetected % The ball was detected.
            % Reduce the measurement noise by calling predict followed by
            % correct.
            predict(kalmanFilter);
            trackedLocation = correct(kalmanFilter, detection);
            label = 'Corrected';
            plot(trackedLocation(1), trackedLocation(2), 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 10);
        else % The ball was missing.
            % Predict the ball's location.
            trackedLocation = predict(kalmanFilter);
            label = 'Predicted';
            plot(trackedLocation(1), trackedLocation(2), 'o', 'MarkerEdgeColor', 'b', 'MarkerSize', 10);
        end
    end
    hold off;
    drawnow
    if ~isempty(trackedLocation)
        KF_Coord = [KF_Coord,[trackedLocation(1);trackedLocation(2)]];
    end
%    F = getframe;
%    writeVideo(vw, F);
end
%close(vw);

%path = 'C:\Users\SakuyasPad\Documents\MATLAB\ECE251B\Project\Evaluation\';
path = [];
save([path, 'KF_Coord'] ,'KF_Coord');