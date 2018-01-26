function loc = computeInitialLocation(param, detectedLocation)
  if strcmp(param.initialLocation, 'Same as first detection')
    loc = detectedLocation;
  else
    loc = param.initialLocation;
  end
end