vr = VideoReader('HandPose_Occu_1.mp4');
load GroundTruth_HandPose_Occu_1_1.mat
i = 1;
while hasFrame(vr)
    CurrentFrame = readFrame(vr);
    figure(1);
    imshow(CurrentFrame);
    hold on;
    loc = GroundTruth(:,i);
    i = i+1;
    plot(loc(1),loc(2),'+','Color','r');
    hold off;
    drawnow;
end