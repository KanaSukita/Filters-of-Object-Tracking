%% Input
function Count = PoseDetector(bboxes, Hand)
if ~isempty(bboxes) && ~isempty(Hand)
    %% Convert bboxes to Centroids
    Fingers = bboxes(:,1:2)+bboxes(:,3:4)./2;
    Hand = Hand(:,1:2)+Hand(:,3:4)./2;
    
    %% Calculate Box
    Left = min(Fingers(:,1));
    Right = max(Fingers(:,1));
    Down = Hand(2);
    Up = min(Fingers(:,2));
    
    Width = abs(Left-Right);
    Height = abs(Up-Down);
    
    %% Count number
    Fingers_Heights = -(Fingers(:,2)-Down);
    TopCount = sum(Fingers_Heights > 0.6.*Height);
    %Fingers_Hori = (Fingers(:,1)-Left);
    %MidCount = sum((Fingers_Hori < 0.1*Width) .* (Fingers_Heights < 0.7.*Height));
    
    Count = TopCount;
else
    Count = 0;
end
end
