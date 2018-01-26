vr = VideoReader('mouse_2.mp4');
vw = VideoWriter('mouse_2_Loss.mp4','MPEG-4');
open(vw);

LossCount = [];
while hasFrame(vr)
    
    Frame = readFrame(vr);
    Loss = 0;
    for j = 1:3
        rd = randi(100);
        if rd >= 80
            if hasFrame(vr)
                Frame = readFrame(vr);
                Loss = Loss+1;
            end
        end
    end
    LossCount = [LossCount, Loss];
    writeVideo(vw, Frame);
end
close(vw);
save('LossCount','LossCount');