function param = getDefaultParameters
  param.motionModel           = 'ConstantAcceleration';
  param.initialLocation       = 'Same as first detection';
  param.initialEstimateError  = 1E5 * ones(1, 3);
  param.motionNoise           = [25, 10, 1];
  param.measurementNoise      = 25;
  param.segmentationThreshold = 0.05;
end