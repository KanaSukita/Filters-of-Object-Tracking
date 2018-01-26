vr = VideoReader('HandPose_Occu_1.mp4');
FrameNum = round(vr.Duration*vr.FrameRate);
GroundTruth = [];
while hasFrame(vr)
    CurrentFrame = readFrame(vr);
    figure(1);
    imshow(CurrentFrame);
    [x, y] = ginput(1);
    [x,y]
    GroundTruth = [GroundTruth, [x;y]];
    FrameNum = FrameNum - 1
end

save('GroundTruth_HandPose_Occu_1_1', 'GroundTruth');