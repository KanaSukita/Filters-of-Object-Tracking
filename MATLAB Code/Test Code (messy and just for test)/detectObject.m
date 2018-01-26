function [detection, isObjectDetected] = detectObject(frame)
  grayImage = rgb2gray(frame);
  utilities.foregroundMask = step(utilities.foregroundDetector, grayImage);
  detection = step(utilities.blobAnalyzer, utilities.foregroundMask);
  if isempty(detection)
    isObjectDetected = false;
  else
    % To simplify the tracking process, only use the first detected object.
    detection = detection(1, :);
    isObjectDetected = true;
  end
end
