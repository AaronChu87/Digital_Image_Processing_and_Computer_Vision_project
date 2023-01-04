function [mapPoints, directionAndDepth, vSetKeyFrames, newPointIdx, stereoMapPointsIndices] = helperLocalBundleAdjustmentStereo(mapPoints, ...
    directionAndDepth, vSetKeyFrames, currKeyFrameId, intrinsics, newPointIdx, stereoMapPointsIndices)
%helperLocalBundleAdjustmentStereo refine the pose of the current key frame and 
%   the map of the surrrounding scene.
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2020-2021 The MathWorks, Inc.

if currKeyFrameId > 2

% Connected key frames of the current key frame
covisViews    = connectedViews(vSetKeyFrames, currKeyFrameId);
covisViewsIds = covisViews.ViewId;

% Identify the fixed key frames that are connected to the connected key frames of the current key frame
fixedViewIds  = [];
for i = 1:numel(covisViewsIds)
    if numel(fixedViewIds) > 10
        break
    end
    tempViews = connectedViews(vSetKeyFrames, covisViewsIds(i));
    tempId    = tempViews.ViewId;
    
    for j = 1:numel(tempId)
        if ~ismember(tempId(j), [fixedViewIds; currKeyFrameId; covisViewsIds])
            fixedViewIds = [fixedViewIds; tempId(j)]; %#ok<AGROW>
            if numel(fixedViewIds) > 10
                break
            end
        end
    end
end

% Always fix the first two key frames
fixedViewIds = [fixedViewIds; intersect(covisViewsIds, [1 2])];

refinedKeyFrameIds = [unique([fixedViewIds; covisViewsIds]); currKeyFrameId];

else
    refinedKeyFrameIds = [1 2];
    fixedViewIds = [1 2];
end

% Refine local key frames and map points
[mapPoints, vSetKeyFrames, mapPointIdx, reprojectionErrors] = bundleAdjustment(...
    mapPoints, vSetKeyFrames, refinedKeyFrameIds, intrinsics, 'FixedViewIDs', fixedViewIds, ...
    'PointsUndistorted', true, 'AbsoluteTolerance', 1e-7,...
    'RelativeTolerance', 1e-16, 'Solver', 'preconditioned-conjugate-gradient', ...
    'MaxIteration', 20);
maxError   = 3; % original is 6
isInlier   = reprojectionErrors < maxError;
inlierIdx  = mapPointIdx(isInlier);
outlierIdx = mapPointIdx(~isInlier);

directionAndDepth = update(directionAndDepth, mapPoints, vSetKeyFrames.Views, inlierIdx, false);
newPointIdx       = setdiff(newPointIdx, outlierIdx);

% Update map points and key frames
if ~isempty(outlierIdx)
    mapPoints = removeWorldPoints(mapPoints, outlierIdx);
    directionAndDepth = remove(directionAndDepth, outlierIdx);
end

newPointIdx = newPointIdx - arrayfun(@(x) nnz(x>outlierIdx), newPointIdx);
stereoMapPointsIndices = stereoMapPointsIndices - arrayfun(@(x) nnz(x>outlierIdx), stereoMapPointsIndices);
end