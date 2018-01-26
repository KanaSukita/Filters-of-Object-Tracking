%% Input
bboxes = [45.1827217302693,366.786604268635,23,48;159.273140802027,215.715190056568,36,44;216.108262770915,208.832962272844,37,46;274.068835933826,230.869881706705,41,59;338.255732777716,297.028262576946,35,56];
Hand = [139,608,158,69];
vr = VideoReader('HandPose.mp4');
Frame = readFrame(vr);
%% Convert bboxes to Centroids
Fingers = bboxes(:,1:2)+bboxes(:,3:4)./2;
Hand = Hand(:,1:2)+Hand(:,3:4)./2;

% figure;
% imshow(Frame);
% hold on;
% for i = 1:size(Centroids,1)
%     cc = Centroids(i,:);
%     plot(cc(1),cc(2), 'o');
% end
% plot(Hand(1),Hand(2),'+');

Left = min(Fingers(:,1));
Right = max(Fingers(:,1));
Down = Hand(2);
Up = min(Fingers(:,2));

Width = abs(Left-Right);
Height = abs(Up-Down);

Fingers_Heights = -(Fingers(:,2)-Down);
TopCount = sum(Fingers_Heights > 0.7.*Height);
Fingers_Hori = (Fingers(:,1)-Left);
MidCount = sum((Fingers_Hori < 0.1*Width) .* (Fingers_Heights < 0.7.*Height));

Count = TopCount+MidCount;
