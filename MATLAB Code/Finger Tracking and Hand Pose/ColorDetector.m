function [detection, bbox, mask] = ColorDetector_Finger(img, RGB_trgt, Threshold, MinPixel, MaxPixel)
H = vision.BlobAnalysis('AreaOutputPort', false, ...
    'MaximumBlobArea', MaxPixel, 'CentroidOutputPort', true, 'MaximumCount', 5);

%% Compare the Color
img_r = img(:,:,1);
img_g = img(:,:,2);
img_b = img(:,:,3);

Diff_r = img_r - RGB_trgt(1);
Diff_g = img_g - RGB_trgt(2);
Diff_b = img_b - RGB_trgt(3);

Diff = (Diff_r.^2 + Diff_g.^2 + Diff_b.^2).^0.5;
Diff = Diff./max(Diff(:));
Matched = Diff < Threshold;
mask = bwareaopen(Matched, MinPixel);
se = strel('disk', 8);
mask = imclose(mask,se);
% [row, col] = find(Matched);
% x = mean(col);
% y = mean(row);
% detection = [x;y];
[detection, bbox] = step(H, mask);
end