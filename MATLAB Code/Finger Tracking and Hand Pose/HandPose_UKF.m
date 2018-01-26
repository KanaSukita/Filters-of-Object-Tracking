%% Set input and parameters
vr = vision.VideoFileReader('HandPose.mp4');
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
nextId = 1;
costOfNonAssignment = 20;
minVisibleCount = 3;
invisibleForTooLong = 5;
ageThreshold = 8;
%f=@(x)[x(1)+x(2)+x(3)/2;x(2)+x(3);x(3);x(4)+x(5)+x(6)/2;x(5)+x(6);x(6)];
f=@(x)[x(1)+x(2);x(2);x(3)+x(4);x(4)];
h=@(x)[x(1);x(3)];
Q=diag([100,25,100,25]);
R=diag([100,100]);
P_ini=diag([200, 50, 200, 50]);

%% Hand Detection
RGB_hand = [0;0;0];
Threshold_hand = 0.15;
MinPixel_hand = 3000;
MaxPixel_hand = 1000000;

%% Finger Detection
RGB_finger = [0;0;0];
Threshold_finger = 0.15;
MinPixel_finger = 400;
MaxPixel_finger = 3000;

%% Main Part
while ~isDone(vr)
    %% Read New Frame
    frame = vr.step();
    
    %% Detect Fingers
    [centroids, bboxes, mask] = ColorDetector_Finger(frame, RGB_finger, Threshold_finger, MinPixel_finger, MaxPixel_finger);
    [~, Hand, ~] = ColorDetector_Hand(frame, RGB_hand, Threshold_hand, MinPixel_hand, MaxPixel_hand);
    bboxes = double(bboxes);
    %% predict New Locations of Tracks & detection To Track Assignment
    nTracks = length(tracks);
    nDetections = size(centroids, 1);
    % Compute the cost of assigning each detection to each track.
    cost = zeros(nTracks, nDetections);
    for i = 1:nTracks
        bbox = tracks(i).bbox;
        % Predict the current location of the track.
        tracks(i).last_state = tracks(i).state;
        tracks(i).last_cov = tracks(i).cov;
        [tracks(i).state, tracks(i).cov, Omiga] = ukf_predict(f,tracks(i).state,tracks(i).cov,h,Q,R);
        %        tracks(i).state = real(f(tracks(i).state));
        x = tracks(i).state;
        Omiga = real(Omiga);
        predictedCentroid = [x(1) x(3)];
        % Shift the bounding box so that its center is at
        % the predicted location.
        predictedCentroid = (predictedCentroid) - bbox(3:4) / 2;
        tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        
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
        [tracks(trackIdx).state, tracks(trackIdx).cov] = ...
            ukf(f,tracks(trackIdx).last_state, tracks(trackIdx).last_cov,h,z,Q,R);
        %[correctedState, ~] = ekf_correct(tracks(trackIdx).state,tracks(trackIdx).cov,h,z,R);
        centroid2 = [tracks(trackIdx).state(1) tracks(trackIdx).state(3)];
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
        state=[centroid(1);0;centroid(2);0];
        cov=P_ini;
        
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
            mask = insertObjectAnnotation(mask, 'rectangle', ...
                Hand, 'Hand', 'Color', 'r');
        end
    end
    
    % Display the mask and the frame.
    figure(1);
    %imshow(mask);
    imshow(frame);
    hold on;
    Count = PoseDetector(bboxes, double(Hand));
    text(240, 100, num2str(Count), 'Color', 'y', 'FontSize', 25);
    hold off;
    drawnow
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
    Y(:,k)=real(f(X(:,k)));
    y=y+Wm(k).*Y(:,k);
end
Y1=Y-y(:,ones(1,L));
P=Y1*diag(Wc)*Y1'+R;

end

%%
%
function [x1,P1,P2]=ukf_predict(fstate,x,P,hmeas,Q,R)
% UKF   Unscented Kalman Filter for nonlinear dynamic systems
% [x, P] = ukf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P
% for nonlinear dynamic system (for simplicity, noises are assumed as additive):

L=numel(x);                                 %numer of states
alpha=1E-3;                                 %default, tunable
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
[z1,Z1,P2,Z2]=ut(hmeas,X1,Wm,Wc,2,R);       %unscented transformation of measurments
%x1 = fstate(x);
end

%%
%
function [x,P]=ukf(fstate,x,P,hmeas,z,Q,R)
% UKF   Unscented Kalman Filter for nonlinear dynamic systems
% [x, P] = ukf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P
% for nonlinear dynamic system (for simplicity, noises are assumed as additive):

L=numel(x);                                 %numer of states
m=numel(z);                                 %numer of measurements
alpha=1E-3;                                 %default, tunable
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
