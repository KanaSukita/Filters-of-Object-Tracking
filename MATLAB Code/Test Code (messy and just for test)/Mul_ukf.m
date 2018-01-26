function MotionBasedMultiObjectTrackingExample()
% Parameters
param.initialEstimateError  = [200,50,50];
param.motionNoise           = [100, 25, 25];
param.measurementNoise      = 10000;
f=@(x)[x(1)+x(2)+x(3)/2;x(2)+x(3);x(3);x(4)+x(5)+x(6)/2;x(5)+x(6);x(6)];  
h=@(x)[x(1);x(4)];                              
Q=diag([param.motionNoise param.motionNoise]);
R=diag([param.measurementNoise param.measurementNoise]);
  
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
 
obj = setupSystemObjects();

tracks = initializeTracks(); % Create an empty array of tracks.

nextId = 1; % ID of the next track

% Detect moving objects, and track them across video frames.
while ~isDone(obj.reader)
    frame = readFrame();
    [centroids, bboxes, mask] = detectObjects(frame);
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();

    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();

    displayTrackingResults();
end

%%
%
    function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.

        % Create a video file reader.
        obj.reader = vision.VideoFileReader('atrium.mp4');

        % Create two video players, one to display the video,
        % and one to display the foreground mask.
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);

        % Create System objects for foreground detection and blob analysis

        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background.

        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);

        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.

        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 400);
    end

%%
%
    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'last_state', {}, ...
            'last_cov', {}, ...
            'state', {}, ...
            'cov', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end

%%
%
    function frame = readFrame()
        frame = obj.reader.step();
    end

%%
%
    function [centroids, bboxes, mask] = detectObjects(frame)

        % Detect foreground.
        mask = obj.detector.step(frame);

        % Apply morphological operations to remove noise and fill in holes.
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');

        % Perform blob analysis to find connected components.
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
    end

%%
%
    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;

            % Predict the current location of the track.
            tracks(i).last_state = tracks(i).state;
            tracks(i).last_cov = tracks(i).cov;
            [tracks(i).state,tracks(i).cov] = ...
                ukf_predict(f,tracks(i).state,tracks(i).cov,Q);
            x=tracks(i).state;
            predictedCentroid = [x(1) x(4)];

            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end

%%
%
    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()

        nTracks = length(tracks);
        nDetections = size(centroids, 1);

        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            x=tracks(i).state;
            cost(i, :) = distance([x(1) x(4)], centroids);
        end

        % Solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
    end

%%
%
    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);

            % Correct the estimate of the object's location
            % using the new detection.
            z=centroid';
            h=@(x)[x(1);x(4)];  
            [tracks(trackIdx).state, tracks(trackIdx).cov] = ...
                ukf(f,tracks(trackIdx).last_state, tracks(trackIdx).last_cov,h,z,Q,R);

            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;

            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

%%
%
    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

%%
%
    function deleteLostTracks()
        if isempty(tracks)
            return;
        end

        invisibleForTooLong = 20;
        ageThreshold = 8;

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end

%%
%
    function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);

        for i = 1:size(centroids, 1)

            centroid = centroids(i,:);
            bbox = bboxes(i, :);

            % Create a Kalman filter object.
            state=[centroid(1);0;0;centroid(2);0;0];
            cov=diag([param.initialEstimateError param.initialEstimateError]);;

            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'last_state', state, ...
                'last_cov', cov, ...
                'state', state, ...
                'cov', cov, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;

            % Increment the next id.
            nextId = nextId + 1;
        end
    end

%%
%
    function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;

        minVisibleCount = 8;
        if ~isempty(tracks)

            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);

                % Get ids.
                ids = int32([reliableTracks(:).id]);

                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);

                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end

        % Display the mask and the frame.
        obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);
    end

%%
%
    function [x1,P1]=ukf_predict(fstate,x,P,Q)
    % UKF   Unscented Kalman Filter for nonlinear dynamic systems
    % [x, P] = ukf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P 
    % for nonlinear dynamic system (for simplicity, noises are assumed as additive):

    L=numel(x);                                 %numer of states
    alpha=1e-3;                                 %default, tunable
    ki=0;                                       %default, tunable
    beta=2;                                     %default, tunable
    lambda=alpha^2*(L+ki)-L;                    %scaling factor
    c=L+lambda;                                 %scaling factor
    Wm=[lambda/c 0.5/c+zeros(1,2*L)];           %weights for means
    Wc=Wm;
    Wc(1)=Wc(1)+(1-alpha^2+beta);               %weights for covariance
    c=sqrt(c);
    X=sigmas(x,P,c);                            %sigma points around x
    [x1,~,P1,~]=ut(fstate,X,Wm,Wc,L,Q);          %unscented transformation of process

    end
%%
%
    function [x,P]=ukf(fstate,x,P,hmeas,z,Q,R)
    % UKF   Unscented Kalman Filter for nonlinear dynamic systems
    % [x, P] = ukf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P 
    % for nonlinear dynamic system (for simplicity, noises are assumed as additive):

    L=numel(x);                                 %numer of states
    m=numel(z);                                 %numer of measurements
    alpha=1e-3;                                 %default, tunable
    ki=0;                                       %default, tunable
    beta=2;                                     %default, tunable
    lambda=alpha^2*(L+ki)-L;                    %scaling factor
    c=L+lambda;                                 %scaling factor
    Wm=[lambda/c 0.5/c+zeros(1,2*L)];           %weights for means
    Wc=Wm;
    Wc(1)=Wc(1)+(1-alpha^2+beta);               %weights for covariance
    c=sqrt(c);
    X=sigmas(x,P,c);                            %sigma points around x
    [x1,X1,P1,X2]=ut(fstate,X,Wm,Wc,L,Q);          %unscented transformation of process
    % X1=sigmas(x1,P1,c);                         %sigma points around x1
    % X2=X1-x1(:,ones(1,size(X1,2)));             %deviation of X1
    [z1,Z1,P2,Z2]=ut(hmeas,X1,Wm,Wc,m,R);       %unscented transformation of measurments
    P12=X2*diag(Wc)*Z2';                        %transformed cross-covariance
    K=P12*inv(P2);
    x=x1+K*(z-z1);                              %state update
    P=P1-K*P12';                                %covariance update
    
    end
%%
%
    function [y,Y,P,Y1]=ut(f,X,Wm,Wc,n,R)
    %Unscented Transformation
    %Input:
    %        f: nonlinear map
    %        X: sigma points
    %       Wm: weights for mean
    %       Wc: weights for covraiance
    %        n: numer of outputs of f
    %        R: additive covariance
    %Output:
    %        y: transformed mean
    %        Y: transformed smapling points
    %        P: transformed covariance
    %       Y1: transformed deviations

    L=size(X,2);
    y=zeros(n,1);
    Y=zeros(n,L);
    for k=1:L                   
        Y(:,k)=f(X(:,k));       
        y=y+Wm(k)*Y(:,k);       
    end
    Y1=Y-y(:,ones(1,L));
    P=Y1*diag(Wc)*Y1'+R;  
    
    end

%%
%
    function X=sigmas(x,P,c)
    %Sigma points around reference point
    %Inputs:
    %       x: reference point
    %       P: covariance
    %       c: coefficient
    %Output:
    %       X: Sigma points
    
    A = c*chol(P)';
    Y = x(:,ones(1,numel(x)));
    X = [x Y+A Y-A]; 
    end 

end