function [xyzPoints, validIndex] = helperReconstructFromRGBD(points, ...
    depthMap, intrinsics, currPose, depthFactor)

if ~isnumeric(points)            % for tracking
    points  = points.Location;   % extract locations from feature points
end

xyzPoints   = zeros(size(points, 1), 3);
maxRange    = 3; % In meters

for i = 1:size(points, 1)
    Z  = double(depthMap(round(points(i, 2)), round(points(i, 1)))) / depthFactor;
    XY = (points(i, :) - intrinsics.PrincipalPoint) ./ intrinsics.FocalLength * Z;
    xyzPoints(i, :)= [XY, Z]; 
end

isPointValid = xyzPoints(:, 3) > 0 & xyzPoints(:, 3) < maxRange;
xyzPoints    = xyzPoints(isPointValid, :);
xyzPoints    = xyzPoints * currPose.Rotation + currPose.Translation;
validIndex   = find(isPointValid);
end