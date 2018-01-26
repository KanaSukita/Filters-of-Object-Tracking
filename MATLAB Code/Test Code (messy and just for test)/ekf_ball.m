function eekf_ball
%%
% Play the video and show the trajectory
%showDetections();

%showTrajectory();

%% 
% Initialization
frame            = [];  % A video frame
detectedLocation = [];  % The detected location
trackedLocation  = [];  % The tracked location
label            = '';  % Label for the ball
utilities        = [];  % Utilities used to process the video

param = getDefaultParameters();  % get Kalman configuration that works well
                                 % for this example
                                 
trackSingleObject(param);  % visualize the results
%%
% The procedure for tracking a single object is shown below.
function trackSingleObject(param)
  % Create utilities used for reading video, detecting moving objects,
  % and displaying the results.
  utilities = createUtilities(param);
  framenum=0;

  isTrackInitialized = false;
  while ~isDone(utilities.videoReader)
    frame = readFrame();

    % Detect the ball.
    [detectedLocation, isObjectDetected] = detectObject(frame);

    if ~isTrackInitialized
      if isObjectDetected
        % Initialize a track by creating a Kalman filter when the ball is
        % detected for the first time.
        initialLocation = computeInitialLocation(param, detectedLocation);
        x=[initialLocation(1);0;0;initialLocation(2);0;0];
        f=@(x)[x(1)+x(2)+x(3)/2;x(2)+x(3);x(3);x(4)+x(5)+x(6)/2;x(5)+x(6);x(6)];  
        h=@(x)[x(1);x(4)];                              
        P=diag([param.initialEstimateError param.initialEstimateError]);
        Q=diag([param.motionNoise param.motionNoise]);
        R=diag([param.measurementNoise param.measurementNoise]);

        isTrackInitialized = true;
        trackedLocation = [x(1) x(4)];
        label = 'Initial';
        
      else
        trackedLocation = [];
        label = '';
      end

    else
      % Use the Kalman filter to track the ball.
      if isObjectDetected % The ball was detected.
        % Reduce the measurement noise by calling predict followed by
        % correct.
        z=[detectedLocation(1);detectedLocation(2)];
        [x, P] = ekf(f,x,P,h,z,Q,R);
        x=real(x);
        trackedLocation = [x(1) x(4)];
        label = 'Corrected';
      else % The ball was missing.
        % Predict the ball's location.
        [x,A]=jaccsd(f,x);    %nonlinear update and linearization at current state
        P=A*P*A'+Q;
        %z=h(x)+sqrt(25)*randn;
        %[x, P] = ekf(f,x,P,h,z,Q,R);
        x=real(x);
        trackedLocation = [x(1) x(4)];
        label = 'Predicted';
      end
    end
    annotateTrackedObject();
  end
  
  showTrajectory();
end

%%
% Get default parameters for creating Kalman filter and for segmenting the
% ball.
function param = getDefaultParameters
  param.motionModel           = 'ConstantAcceleration';
  param.initialLocation       = 'Same as first detection';
  param.initialEstimateError  = 1E5 * ones(1, 3);
  param.motionNoise           = [25, 10, 1];
  param.measurementNoise      = 25;
  param.segmentationThreshold = 0.05;
end

%%
% Read the next video frame from the video file.
function frame = readFrame()
  frame = step(utilities.videoReader);
end

%%
% Detect and annotate the ball in the video.
function showDetections()
  param = getDefaultParameters();
  utilities = createUtilities(param);
  trackedLocation = [];

  idx = 0;
  while ~isDone(utilities.videoReader)
    frame = readFrame();
    detectedLocation = detectObject(frame);
    % Show the detection result for the current video frame.
    annotateTrackedObject();

    % To highlight the effects of the measurement noise, show the detection
    % results for the 40th frame in a separate figure.
    idx = idx + 1;
    if idx == 40
      combinedImage = max(repmat(utilities.foregroundMask, [1,1,3]), frame);
      figure, imshow(combinedImage);
    end
  end % while
  
  % Close the window which was used to show individual video frame.
  uiscopes.close('All'); 
end

%%
% Detect the ball in the current video frame.
function [detection, isObjectDetected] = detectObject(frame)
  grayImage = rgb2gray(frame);
  utilities.foregroundMask = step(utilities.foregroundDetector, grayImage);
  detection = step(utilities.blobAnalyzer, utilities.foregroundMask);
  if isempty(detection)
    isObjectDetected = false;
  else
    % To simplify the tracking process, only use the first detected object.
    detection = detection(1, :);
    isObjectDetected = true;
  end
end

%%
% Show the current detection and tracking results.
function annotateTrackedObject()
  accumulateResults();
  % Combine the foreground mask with the current video frame in order to
  % show the detection result.
  combinedImage = max(repmat(utilities.foregroundMask, [1,1,3]), frame);

  if ~isempty(trackedLocation)
    shape = 'circle';
    region = trackedLocation;
    region(:, 3) = 5;
    combinedImage = insertObjectAnnotation(combinedImage, shape, ...
      region, {label}, 'Color', 'red');
  end
  step(utilities.videoPlayer, combinedImage);
end

%%
% Show trajectory of the ball by overlaying all video frames on top of 
% each other.
function showTrajectory
  % Close the window which was used to show individual video frame.
  uiscopes.close('All'); 
  
  % Create a figure to show the processing results for all video frames.
  figure; imshow(utilities.accumulatedImage/2+0.5); hold on;
  plot(utilities.accumulatedDetections(:,1), ...
    utilities.accumulatedDetections(:,2), 'k+');
  
  if ~isempty(utilities.accumulatedTrackings)
    plot(utilities.accumulatedTrackings(:,1), ...
      utilities.accumulatedTrackings(:,2), 'r-o');
    legend('Detection', 'Tracking');
  end
end

%%
% Accumulate video frames, detected locations, and tracked locations to
% show the trajectory of the ball.
function accumulateResults()
  utilities.accumulatedImage      = max(utilities.accumulatedImage, frame);
  utilities.accumulatedDetections ...
    = [utilities.accumulatedDetections; detectedLocation];
  utilities.accumulatedTrackings  ...
    = [utilities.accumulatedTrackings; trackedLocation];
end

%%
% For illustration purposes, select the initial location used by the Kalman
% filter.
function loc = computeInitialLocation(param, detectedLocation)
  if strcmp(param.initialLocation, 'Same as first detection')
    loc = detectedLocation;
  else
    loc = param.initialLocation;
  end
end

%%
% Create utilities for reading video, detecting moving objects, and
% displaying the results.
function utilities = createUtilities(param)
  % Create System objects for reading video, displaying video, extracting
  % foreground, and analyzing connected components.
  utilities.videoReader = vision.VideoFileReader('singleball.mp4');
  utilities.videoPlayer = vision.VideoPlayer('Position', [100,100,500,400]);
  utilities.foregroundDetector = vision.ForegroundDetector(...
    'NumTrainingFrames', 10, 'InitialVariance', param.segmentationThreshold);
  utilities.blobAnalyzer = vision.BlobAnalysis('AreaOutputPort', false, ...
    'MinimumBlobArea', 70, 'CentroidOutputPort', true);

  utilities.accumulatedImage      = 0;
  utilities.accumulatedDetections = zeros(0, 2);
  utilities.accumulatedTrackings  = zeros(0, 2);
end

displayEndOfDemoMessage(mfilename)

%%
function [z,A]=jaccsd(fun,x)
  z=fun(x);
  n=numel(x);
  m=numel(z);
  A=zeros(m,n);
  h=n*eps;
  for k=1:n
    x1=x;
    x1(k)=x1(k)+h*i;
    A(:,k)=imag(fun(x1))/h;
  end
end
%%
function [x,P]=ekf(fstate,x,P,hmeas,z,Q,R)
% EKF   Extended Kalman Filter for nonlinear dynamic systems
% [x, P] = ekf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P 
% for nonlinear dynamic system:
%           x_k+1 = f(x_k) + w_k
%           z_k   = h(x_k) + v_k
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
% Inputs:   f: function handle for f(x)
%           x: "a priori" state estimate
%           P: "a priori" estimated state covariance
%           h: fanction handle for h(x)
%           z: current measurement
%           Q: process noise covariance 
%           R: measurement noise covariance
% Output:   x: "a posteriori" state estimate
%           P: "a posteriori" state covariance
  [x1,A]=jaccsd(fstate,x);    %nonlinear update and linearization at current state
  P=A*P*A'+Q;                 %partial update
  [z1,H]=jaccsd(hmeas,x1);    %nonlinear measurement and linearization
  P12=P*H';                   %cross covariance
  R=chol(H*P12+R);            %Cholesky factorization
  U=P12/R;                    %K=U/R'; Faster because of back substitution
  x=x1+U*(R'\(z-z1));         %Back substitution to get state update
  P=P-U*U';                   %Covariance update, U*U'=P12/R/R'*P12'=K*P12.

function [z,A]=jaccsd(fun,x)
  z=fun(x);
  n=numel(x);
  m=numel(z);
  A=zeros(m,n);
  h=n*eps;
  for k=1:n
    x1=x;
    x1(k)=x1(k)+h*i;
    A(:,k)=imag(fun(x1))/h;
  end
end

end

end