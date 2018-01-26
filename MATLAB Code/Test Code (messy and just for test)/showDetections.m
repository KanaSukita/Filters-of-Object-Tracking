function showDetections()
  param = getDefaultParameters();
  utilities = createUtilities(param);
  trackedLocation = [];

  idx = 0;
  while ~isDone(utilities.videoReader)
    frame = readFrame();
    detectedLocation = detectObject(frame);
    % Show the detection result for the current video frame.
    annotateTrackedObject();

    % To highlight the effects of the measurement noise, show the detection
    % results for the 40th frame in a separate figure.
    idx = idx + 1;
    if idx == 40
      combinedImage = max(repmat(utilities.foregroundMask, [1,1,3]), frame);
      figure, imshow(combinedImage);
    end
  end % while

  % Close the window which was used to show individual video frame.
  uiscopes.close('All');
end

