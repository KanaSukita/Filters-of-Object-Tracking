function MotionBasedMultiObjectTrackingExample()
% Parameters
param.initialEstimateError  = [200,50,50];
param.motionNoise           = [100, 25, 25];
param.measurementNoise      = 250000;
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
            [tracks(i).state, tracks(i).cov] = ...
                ekf_predict(f,tracks(i).state,tracks(i).cov,Q);
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
                ekf_correct(tracks(trackIdx).state,tracks(trackIdx).cov,h,z,R);

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
            cov=diag([param.initialEstimateError param.initialEstimateError]);

            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
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
    function [z,A]=jaccsd(fun,x)
        % JACCSD Jacobian through complex step differentiation
        % [z J] = jaccsd(f,x)
        % z = f(x)
        % J = f'(x)
        %
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
%
    function [x,P]=ekf_predict(fstate,x,P,Q)
        [x,A]=jaccsd(fstate,x);    %nonlinear update and linearization at current state
        P=A*P*A'+Q; 
    end

%%
%
    function [x,P]=ekf_correct(x,P,hmeas,z,R)
        [z1,H]=jaccsd(hmeas,x);    %nonlinear measurement and linearization
        P12=P*H';                   %cross covariance
        % K=P12*inv(H*P12+R);       %Kalman filter gain
        % x=x1+K*(z-z1);            %state estimate
        % P=P-K*P12';               %state covariance matrix
        R=chol(H*P12+R);            %Cholesky factorization
        U=P12/R;                    %K=U/R'; Faster because of back substitution
        x=x+U*(R'\(z-z1));         %Back substitution to get state update
        P=P-U*U';                   %Covariance update, U*U'=P12/R/R'*P12'=K*P12.
    end
end