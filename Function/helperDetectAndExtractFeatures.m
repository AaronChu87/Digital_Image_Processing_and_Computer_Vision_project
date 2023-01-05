function [features, validPoints] = helperDetectAndExtractFeatures(Irgb, ContrastThreshold, EdgeThreshold, Sigma, Layers, numPoints)
 
% numPoints = 1000; % setting the number of feature points

% Detect SIFT features
Igray  = rgb2gray(Irgb);

points = detectSIFTFeatures(Igray, 'ContrastThreshold', ContrastThreshold, 'EdgeThreshold', EdgeThreshold, 'NumLayersInOctave', Layers, 'Sigma', Sigma);

% Select a subset of features, uniformly distributed throughout the image
% points = selectUniform(points, numPoints, size(Igray, 1:2));

% Extract features
[features, validPoints] = extractFeatures(Igray, points);
features = binaryFeatures(uint8(features));
end