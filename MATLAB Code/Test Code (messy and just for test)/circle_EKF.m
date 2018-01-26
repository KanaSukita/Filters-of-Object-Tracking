%% Set input and parameters
vr = vision.VideoFileReader('white_circle_occ.mp4');
tracks = struct(...
    'id', {}, ...
    'bbox', {}, ...
    'state', {}, ...
    'cov', {}, ...
    'age', {}, ...
    'totalVisibleCount', {}, ...
    'consecutiveInvisibleCount', {});
nextId = 1;
costOfNonAssignment = 20;
minVisibleCount = 3;
invisibleForTooLong = 20;
ageThreshold = 8;
%f=@(x)[x(1)+x(2)+x(3)/2;x(2)+x(3);x(3);x(4)+x(5)+x(6)/2;x(5)+x(6);x(6)];
f = @(x)[300+100*cos((-1)^(x(2)-200<0)*acos((x(1)-300)/100)+x(3));...
    200+100*sin((-1)^(x(2)-200<0)*acos((x(1)-300)/100)+x(3));x(3)];
h=@(x)[x(1);x(2)];
Q=diag([10,10,0.0004]);
R=diag([100,100]);
P_ini=diag([200, 200, 50]);

%% Hand Detection
RGB_hand = [1;1;1];
Threshold_hand = 0.35;
MinPixel_hand = 500;
MaxPixel_hand = 1000000;

%% Finger Detection
RGB_finger = [1;1;1];
Threshold_finger = 0.12;
MinPixel_finger = 1;
MaxPixel_finger = 600;

%% Main Part
while ~isDone(vr)
    %% Read New Frame
    frame = vr.step();
    
    %% Detect Fingers
    [centroids, bboxes, mask] = ColorDetector(frame, RGB_finger, Threshold_finger, MinPixel_finger, MaxPixel_finger);
    bboxes = double(bboxes);
    %% predict New Locations of Tracks & detection To Track Assignment
    nTracks = length(tracks);
    nDetections = size(centroids, 1);
    % Compute the cost of assigning each detection to each track.
    cost = zeros(nTracks, nDetections);
    for i = 1:nTracks
        bbox = tracks(i).bbox;
        % Predict the current location of the track.
        [tracks(i).state, tracks(i).cov] = ekf_predict(f,tracks(i).state,tracks(i).cov,Q);
        tracks(i).state = real(tracks(i).state);
        x=tracks(i).state;
        predictedCentroid = [x(1) x(2)];
        % Shift the bounding box so that its center is at
        % the predicted location.
        predictedCentroid = (predictedCentroid) - bbox(3:4) / 2;
        tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        
        Omiga = getOmiga(tracks(i).state,tracks(i).cov,h,R);
        for j = 1:nDetections
            cost(i,j) = (centroids(j,:)-h(x)')*inv(Omiga)*(centroids(j,:)-h(x)')' + log(det(Omiga));
        end
    end
    % Solve the assignment problem.
    [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost, costOfNonAssignment);
    
    %% update Assigned Tracks
    numAssignedTracks = size(assignments, 1);
    for i = 1:numAssignedTracks
        trackIdx = assignments(i, 1);
        detectionIdx = assignments(i, 2);
        centroid = centroids(detectionIdx, :);
        bbox = bboxes(detectionIdx, :);
        
        % Correct the estimate of the object's location
        % using the new detection.
        z=centroid';
        [tracks(trackIdx).state, tracks(trackIdx).cov] = ekf_correct(tracks(trackIdx).state,tracks(trackIdx).cov,h,z,R);
        %[correctedState, ~] = ekf_correct(tracks(trackIdx).state,tracks(trackIdx).cov,h,z,R);
        centroid2 = [tracks(trackIdx).state(1) tracks(trackIdx).state(2)];
        Diff = centroid2 - centroid;
        bbox(1:2) = bbox(1:2) + Diff;
        % Replace predicted bounding box with detected
        % bounding box.
        tracks(trackIdx).bbox = bbox;
        
        % Update track's age.
        tracks(trackIdx).age = tracks(trackIdx).age + 1;
        
        % Update visibility.
        tracks(trackIdx).totalVisibleCount = tracks(trackIdx).totalVisibleCount + 1;
        tracks(trackIdx).consecutiveInvisibleCount = 0;
    end
    
    %% update Unassigned Tracks
    for i = 1:length(unassignedTracks)
        ind = unassignedTracks(i);
        tracks(ind).age = tracks(ind).age + 1;
        tracks(ind).consecutiveInvisibleCount = ...
            tracks(ind).consecutiveInvisibleCount + 1;
    end
    
    %% deleteLostTracks()
    if ~isempty(tracks)
        
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
    %% Create New Tracks
    centroids = centroids(unassignedDetections, :);
    bboxes = bboxes(unassignedDetections, :);
    
    for i = 1:size(centroids, 1)
        centroid = centroids(i,:);
        bbox = bboxes(i, :);
        % Create a Kalman filter object.
        state=[centroid(1);centroid(2);0];
        cov=P_ini;
        
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
    %% display Tracking Results
    frame = im2uint8(frame);
    mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
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
            predictedTrackInds = [reliableTracks(:).consecutiveInvisibleCount] > 0;
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
    figure(1);
    imshow(mask);
    %    figure(2);
    %    imshow(frame);
end

%%
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
    x1(k)=x1(k)+h*1i;
    A(:,k)=imag(fun(x1))/h;
end
end

%%
function [x,P]=ekf_predict(fstate,x,P,Q)
[x,A]=jaccsd(fstate,x);    %nonlinear update and linearization at current state
P=A*P*A'+Q;
end

%%
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

%% getOmiga
function omiga = getOmiga(x,P,hmeas,R)
[~,H]=jaccsd(hmeas,x);
omiga = H*P*H'+R;
end