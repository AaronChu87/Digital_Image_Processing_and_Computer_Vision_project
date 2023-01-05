clc;clear;close all;rng(0);
addpath("Function\")
%%
imageFolder = 'rgbd_dataset_freiburg3_long_office_household\';
imgFolderColor = [imageFolder,'rgb/'];
imgFolderDepth = [imageFolder,'depth/'];
imdsColor      = imageDatastore(imgFolderColor);
imdsDepth      = imageDatastore(imgFolderDepth);
% Load time stamp data of color images
timeColor = helperImportTimestampFile([imageFolder, 'rgb.txt']);

% Load time stamp data of depth images
timeDepth = helperImportTimestampFile([imageFolder, 'depth.txt']);

% Align the time stamp
indexPairs = helperAlignTimestamp(timeColor, timeDepth);

% Select the synchronized image data
imdsColor     = subset(imdsColor, indexPairs(:, 1));
imdsDepth     = subset(imdsDepth, indexPairs(:, 2));
%% First frame
currFrameIdx  = 1;
currIcolor    = readimage(imdsColor,currFrameIdx);
currIdepth    = readimage(imdsDepth,currFrameIdx);
% imshowpair(currIcolor, currIdepth, 'montage');
%%
% Set random seed for reproducibility
rng(0);

% Create a cameraIntrinsics object to store the camera intrinsic parameters.
% The intrinsics for the dataset can be found at the following page:
% https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
focalLength    = [535.4, 539.2];    % in units of pixels
principalPoint = [320.1, 247.6];    % in units of pixels
imageSize      = size(currIcolor,[1,2]); % in pixels [mrows, ncols]
depthFactor    = 5e3;
intrinsics     = cameraIntrinsics(focalLength,principalPoint,imageSize);

% Detect and extract ORB features from the color image
ContrastThreshold = 0.0133;
EdgeThreshold = 10;
Sigma = 1.6;
Layers = 3;
numPoints = 1000;
scaleFactor = 1.2;
numLevels = 8;
% Detect and extract SIFT features from the color image
[currFeatures, currPoints] = helperDetectAndExtractFeatures(currIcolor, ContrastThreshold, EdgeThreshold, Sigma, Layers, numPoints);

initialPose = rigid3d(); % Pose of the first frame, the origin of the map
[xyzPoints, validIndex] = helperReconstructFromRGBD(currPoints, currIdepth, intrinsics, initialPose, depthFactor);

%% Data Management and Visualization
currKeyFrameId = 1;
% Create an empty imageviewset object to store key frames
vSetKeyFrames = imageviewset;

% Create an empty worldpointset object to store 3-D map sparse feature points
mapPointSet   = worldpointset;

% Create a helperViewDirectionAndDepth object to store view direction and depth 
directionAndDepth = helperViewDirectionAndDepth(size(xyzPoints, 1));

% Add the first key frame
vSetKeyFrames = addView(vSetKeyFrames, currKeyFrameId, initialPose, 'Points', currPoints,...
    'Features', currFeatures.Features);

% Add 3-D map points
[mapPointSet, rgbdMapPointsIdx] = addWorldPoints(mapPointSet, xyzPoints);

% Add observations of the map points
mapPointSet = addCorrespondences(mapPointSet, currKeyFrameId, rgbdMapPointsIdx, validIndex);

% Visualize matched features in the first key frame
featurePlot = helperVisualizeMatchedFeaturesRGBD(currIcolor, currIdepth, currPoints(validIndex));
% Visualize initial map points and camera trajectory
xLim = [-3 3];
yLim = [-3 2];
zLim = [-1 5];
mapPlot  = helperVisualizeMotionAndStructure(vSetKeyFrames, mapPointSet, xLim, yLim, zLim);
title('TUM Dataset','Color','w','FontSize',20,'Parent',mapPlot.Axes)
% vobj.writeVideo(getframe(mapPlot.Axes.Parent));
%% Tracking
% ViewId of the last key frame
lastKeyFrameId    = currKeyFrameId;

% Index of the last key frame in the input image sequence
lastKeyFrameIdx   = currFrameIdx; 

% Indices of all the key frames in the input image sequence
addedFramesIdx    = lastKeyFrameIdx;

currFrameIdx      = currFrameIdx + 1;

isLastFrameKeyFrame = true;
%% Main loop
while currFrameIdx < numel(imdsColor.Files)
    currIcolor = readimage(imdsColor,currFrameIdx);
    currIdepth = readimage(imdsDepth,currFrameIdx);

    [currFeatures, currPoints]    = helperDetectAndExtractFeatures(currIcolor, ContrastThreshold, EdgeThreshold, Sigma, Layers, numPoints);

    % Track the last key frame
    % trackedMapPointsIdx:  Indices of the map points observed in the current left frame 
    % trackedFeatureIdx:    Indices of the corresponding feature points in the current left frame
    [currPose, trackedMapPointsIdx, trackedFeatureIdx] = helperTrackLastKeyFrame(mapPointSet, ...
        vSetKeyFrames.Views, currFeatures, currPoints, lastKeyFrameId, intrinsics, scaleFactor);
    
    if isempty(currPose) || numel(trackedMapPointsIdx) < 30  
        % current frame can not find enough matching feature or too different with to previous keyframe
        currFrameIdx = currFrameIdx + 1;
        continue
    end

    numSkipFrames     = 20;
    numPointsKeyFrame = 100;
    [localKeyFrameIds, currPose, trackedMapPointsIdx, trackedFeatureIdx, isKeyFrame] = ...
        helperTrackLocalMap(mapPointSet, directionAndDepth, vSetKeyFrames, trackedMapPointsIdx, ...
        trackedFeatureIdx, currPose, currFeatures, currPoints, intrinsics, scaleFactor, numLevels, ...
        isLastFrameKeyFrame, lastKeyFrameIdx, currFrameIdx, numSkipFrames, numPointsKeyFrame);
    % In 'helperTrackLocalMap', currPose is BA with reference keyframe. 

    % Match feature points between the stereo images and get the 3-D world positions
    [xyzPoints, validIndex] = helperReconstructFromRGBD(currPoints, currIdepth, ...
        intrinsics, currPose, depthFactor);

    % Visualize matched features
    updatePlot(featurePlot, currIcolor, currIdepth, currPoints(trackedFeatureIdx));
    
    if ~isKeyFrame %currFrameIdx ~= size(Depth,3)
        currFrameIdx = currFrameIdx + 1;
        isLastFrameKeyFrame = false;
        continue
    else
        [untrackedFeatureIdx, ia] = setdiff(validIndex, trackedFeatureIdx);
        xyzPoints = xyzPoints(ia, :);   % new 3D points in keyframe, world map
        isLastFrameKeyFrame = true;
    end

    % Update current key frame ID
    currKeyFrameId  = currKeyFrameId + 1;

%% Local mapping  (stereo)
    % Add the new key frame    
    % add correspondence
    [mapPointSet, vSetKeyFrames] = helperAddNewKeyFrame(mapPointSet, vSetKeyFrames, ...
        currPose, currFeatures, currPoints, trackedMapPointsIdx, trackedFeatureIdx, localKeyFrameIds); 
        
    % Remove outlier map points that are observed in fewer than 3 key frames
    if currKeyFrameId == 2
        triangulatedMapPointsIdx = [];
    end
    
    [mapPointSet, directionAndDepth, trackedMapPointsIdx] = ...
        helperCullRecentMapPoints(mapPointSet, directionAndDepth, trackedMapPointsIdx, triangulatedMapPointsIdx, ...
        rgbdMapPointsIdx);

    % Add new map points computed from disparity 
    [mapPointSet, rgbdMapPointsIdx] = addWorldPoints(mapPointSet, xyzPoints);
    mapPointSet = addCorrespondences(mapPointSet, currKeyFrameId, rgbdMapPointsIdx, ...
        untrackedFeatureIdx);
    
    % Create new map points by triangulation
    minNumMatches = 10;
    minParallax   = 0.35;
    [mapPointSet, vSetKeyFrames, triangulatedMapPointsIdx, rgbdMapPointsIdx] = helperCreateNewMapPointsStereo( ...
        mapPointSet, vSetKeyFrames, currKeyFrameId, intrinsics, scaleFactor, minNumMatches, minParallax, ...
        untrackedFeatureIdx, rgbdMapPointsIdx);
    
    % Update view direction and depth
    directionAndDepth = update(directionAndDepth, mapPointSet, vSetKeyFrames.Views, ...
        [trackedMapPointsIdx; triangulatedMapPointsIdx; rgbdMapPointsIdx], true);
    
    % Local bundle adjustment
    [mapPointSet, directionAndDepth, vSetKeyFrames, triangulatedMapPointsIdx, rgbdMapPointsIdx] = ...
        helperLocalBundleAdjustmentStereo(mapPointSet, directionAndDepth, vSetKeyFrames, ...
        currKeyFrameId, intrinsics, triangulatedMapPointsIdx, rgbdMapPointsIdx); 
    
    % Visualize 3-D world points and camera trajectory
    updatePlot(mapPlot, vSetKeyFrames, mapPointSet);

    lastKeyFrameId  = currKeyFrameId;
    lastKeyFrameIdx = currFrameIdx;
    addedFramesIdx  = [addedFramesIdx; currFrameIdx]; %#ok<AGROW>
    currFrameIdx    = currFrameIdx + 1;
end
%% Plot estimate camera
for i = 1:numel(addedFramesIdx)
    plotCamera("AbsolutePose",vSetKeyFrames.Views.AbsolutePose(i),"AxesVisible",false,"Color","red",'Size', 0.05, Parent=mapPlot.Axes)
end
%% Plot ground truth
gTruthData = load("Ground_truth\Human_Ground_truth.mat");
gTruth     = gTruthData.gTruth;
GTruth = vertcat(gTruth.Translation);
% traj = line([GTruth(1:end,1)],[GTruth(1:end,2)],[GTruth(1:end,3)],"LineWidth",3,"Color",[0,1,0],"LineStyle","-", Parent=mapPlot.Axes);
plot3(GTruth(addedFramesIdx,1), GTruth(addedFramesIdx,2), GTruth(addedFramesIdx,3),'-o','LineWidth',1,'Color',[0 1 0], 'DisplayName', 'Actual trajectory', Parent=mapPlot.Axes);
showLegend(mapPlot);

%% pose graph
% OptimizedPoses = dense_pose_graph_2(vSetKeyFrames);
OptimizedPoses = dense_pose_graph(vSetKeyFrames, mapPointSet, intrinsics);
pose_graph_opt = [];
for i = 1:numel(addedFramesIdx)
    optpose = updateView(vSetKeyFrames,i,rigid3d(OptimizedPoses{i}(1:3,1:3)',OptimizedPoses{i}(1:3,4)'));
    pose_graph_opt = [pose_graph_opt;OptimizedPoses{i}(1:3,4)'];
end
plot3(pose_graph_opt(:,1), pose_graph_opt(:,2), pose_graph_opt(:,3),'b','LineWidth',2, 'DisplayName', 'Actual trajectory', Parent=mapPlot.Axes);
%% Dense Reconstruction from Depth Image
% Create an array of pointCloud objects to store the world points constructed
% from the key frames
ptClouds =  repmat(pointCloud(zeros(1, 3)), numel(addedFramesIdx), 1);

% Ignore image points at the boundary 
offset = 40;
[X, Y] = meshgrid(offset:2:imageSize(2)-offset, offset:2:imageSize(1)-offset);
OptimizedPoses  = poses(vSetKeyFrames);
for i = 1: numel(addedFramesIdx)
    Icolor = readimage(imdsColor,addedFramesIdx(i));
    Idepth = readimage(imdsDepth,addedFramesIdx(i));
% 
    [xyzPoints, validIndex] = helperReconstructFromRGBD([X(:), Y(:)], ...
        Idepth, intrinsics, OptimizedPoses.AbsolutePose(i), depthFactor);
%     [xyzPoints, validIndex] = helperReconstructFromRGBD([X(:), Y(:)], ...
%         Idepth, intrinsics, rigid3d(OptimizedPoses{i}(1:3,1:3)',OptimizedPoses{i}(1:3,4)'), depthFactor);

    colors = zeros(numel(X), 1, 'like', Icolor);
    for j = 1:numel(X)
        colors(j, 1:3) = Icolor(Y(j), X(j), :);
    end
    ptClouds(i) = pointCloud(xyzPoints, 'Color', colors(validIndex, :));
end

% Concatenate the point clouds
pointCloudsAll = pccat(ptClouds);

figure
pcshow(pointCloudsAll,'VerticalAxis', 'y', 'VerticalAxisDir', 'down');
xlabel('X')
ylabel('Y')
zlabel('Z')

%% calculate error
BAPoses  = poses(vSetKeyFrames);
DATA_LEN = size(addedFramesIdx,1);
t_ba = zeros(DATA_LEN,3);
for i = 1:DATA_LEN
    t_ba(i,:) = BAPoses{i,2}.Translation;
end
t_gt = GTruth(addedFramesIdx,:);
%
error_t_ba = sqrt(sum((t_ba-t_gt).^2,2));

figure
plot(error_t_ba,'LineWidth',2,'Color','blue')
grid on
title('Trajactory error','FontSize',16)
xlabel('Frame','FontSize',12)
ylabel('Error (m)','FontSize',12)
axis([1 DATA_LEN 0 inf])
