clc;clear;close all;
%%
imageFolder = 'D:\rgbd_dataset_freiburg3_long_office_household\rgbd_dataset_freiburg3_long_office_household\';
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
%% Initialization
focalLength    = [535.4, 539.2];    % in units of pixels
principalPoint = [320.1, 247.6];    % in units of pixels
imageSize      = size(currIcolor,[1,2]); % in pixels [mrows, ncols]
depthFactor    = 5e3;
intrinsics     = cameraIntrinsics(focalLength,principalPoint,imageSize);
%% Plot ground truth
load("TUM_KeyFramesIdx.mat")
gTruthData = load('orbslamGroundTruth.mat');
gTruth     = gTruthData.gTruth;
gTruth = gTruth(indexPairs(addedFramesIdx, 1));
% GTruth = vertcat(gTruth.Translation);
%%
n = size(gTruth,2);
% t = 20;
% k = round(n/t);
%% Plot ground truth
% pose_temp = cell(size(gTruth,1),1);
% pose_gt = cell(k,1);
% trans_gt = zeros(k,3);
% for i = 1:k
%     pose_gt{i} = gTruth(i);
%     trans_gt(i,:) = gTruth(i).Translation;
% end
%%
ptClouds =  repmat(pointCloud(zeros(1, 3)), n, 1);
% Ignore image points at the boundary 
offset = 40;
[X, Y] = meshgrid(offset:2:imageSize(2)-offset, offset:2:imageSize(1)-offset);

for i = 1: n
    Icolor = readimage(imdsColor,addedFramesIdx(i));
    Idepth = readimage(imdsDepth,addedFramesIdx(i));

    [xyzPoints, validIndex] = helperReconstructFromRGBD([X(:), Y(:)], ...
        Idepth, intrinsics, gTruth(i), depthFactor);

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

% mesh = pc2surfacemesh(pointCloudsAll,"poisson",8);
% surfaceMeshShow(mesh)