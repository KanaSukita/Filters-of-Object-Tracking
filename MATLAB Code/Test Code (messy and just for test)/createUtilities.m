function utilities = createUtilities(param)
  % Create System objects for reading video, displaying video, extracting
  % foreground, and analyzing connected components.
  utilities.videoReader = vision.VideoFileReader('singleball.mp4');
  utilities.videoPlayer = vision.VideoPlayer('Position', [100,100,500,400]);
  utilities.foregroundDetector = vision.ForegroundDetector(...
    'NumTrainingFrames', 10, 'InitialVariance', param.segmentationThreshold);
  utilities.blobAnalyzer = vision.BlobAnalysis('AreaOutputPort', false, ...
    'MinimumBlobArea', 70, 'CentroidOutputPort', true);

  utilities.accumulatedImage      = 0;
  utilities.accumulatedDetections = zeros(0, 2);
  utilities.accumulatedTrackings  = zeros(0, 2);
end