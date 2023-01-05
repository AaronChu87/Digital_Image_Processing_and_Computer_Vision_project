function [Poses] = dense_pose_graph(vSetKeyFrames, mapPointSet, intrinsics)
%% Parameters
w_loop = 10;
threshold = 0.0001;
numKeyframe = vSetKeyFrames.NumViews;
%% record keyframe pose
t = zeros(numKeyframe,3);
T = cell(numKeyframe,1);
for i = 1:numKeyframe
    t(i,:) = vSetKeyFrames.Views.AbsolutePose(i).Translation;
    T{i} = [vSetKeyFrames.Views.AbsolutePose(i).Rotation', vSetKeyFrames.Views.AbsolutePose(i).Translation'; 0 0 0 1];
end
%% Pose edge
for i = 1:numKeyframe-1
    edge_store{i,1} = [i,i+1];
    pose_front = vSetKeyFrames.Views.AbsolutePose(i);
    R_front = pose_front.Rotation';
    t_front = pose_front.Translation';

    pose_end = vSetKeyFrames.Views.AbsolutePose(i+1);
    R_end = pose_end.Rotation';
    t_end = pose_end.Translation';

    delta_T = [R_end, t_end;0 0 0 1]^-1*[R_front, t_front;0 0 0 1];
    
%     delta_T_rigid3d = rigid3d(delta_T(1:3,1:3),delta_T(1:3,4)');
%     edge_store{i,2} = delta_T_rigid3d;
    edge_store{i,2} = delta_T;
end
% edge_store{numKeyframe,1} = [numKeyframe,1];
% edge_store{numKeyframe,2} = eye(4);
%% loop edge
match_times = 0;
for i = 1:numKeyframe-2
    k = 2;
    [pointId_front, ~] = findWorldPointsInView(mapPointSet,i);
    
    while i+k <= numKeyframe
        viewId = [i,i+k];

        [pointId_end, featureId_end] = findWorldPointsInView(mapPointSet,i+k);
        [covis_pt, ~, ib] = intersect(pointId_front, pointId_end);
        % check numbers of covisble points
        if size(covis_pt,1) < 50
            break
        end

        % poses from i to i+k 
        pose_front = vSetKeyFrames.Views.AbsolutePose(i);
        R_front = pose_front.Rotation';
        t_front = pose_front.Translation';
        T_front_C2W = [R_front, t_front;0 0 0 1];
        pose_end = vSetKeyFrames.Views.AbsolutePose(i+k);
        R_end = pose_end.Rotation';
        t_end = pose_end.Translation';
        
        delta_T_C2W = [R_end, t_end;0 0 0 1]^-1*[R_front, t_front;0 0 0 1];
        delta_T = rigid3d(delta_T_C2W(1:3,1:3)',delta_T_C2W(1:3,4)');

        % covisible 3D points
        pt_3D = mapPointSet.WorldPoints(covis_pt,:);

        % project 3D points to frame i
        T_front_W2C = T_front_C2W^-1;
        pt_3D_i = (T_front_W2C (1:3,1:3)*pt_3D'+T_front_W2C (1:3,4))';

        % covisible feature poins in i+k frames
        covis_ft_idx = featureId_end(ib);
        pt_2D = vSetKeyFrames.Views.Points{i+k}.Location(covis_ft_idx,:);  % 
            
        [BAPose, err] = bundleAdjustmentMotion(pt_3D_i, pt_2D, delta_T, intrinsics, 'PointsUndistorted', true, ...
        'AbsoluteTolerance', 1e-7, 'RelativeTolerance', 1e-16,'MaxIteration', 20,'Verbose',false);

        if err > 10
            continue
        end
        match_times = match_times+1;

        T_C2W = [BAPose.Rotation', BAPose.Translation';0 0 0 1];
        % save pose
        edge_store{numKeyframe + match_times -1,1} = viewId;
        edge_store{numKeyframe + match_times -1,2} = T_C2W^-1;

        k = k+1;
    end    
end
% loop edge last to 1

%%
infoi = [1 2 3 4 5 6];
infoj = [1 2 3 4 5 6];
mu = 1;
pose_num = numKeyframe;
edge_num = size(edge_store,1);
%%
for iter1 = 1:100
    H = sparse(pose_num*6,pose_num*6); 
    b = sparse(pose_num*6,1);
    
    for iter2 = 1 : edge_num
        i = edge_store{iter2,1}(1); 
        j = edge_store{iter2,1}(2);
        if j-1 == i
                info = [1000 1000 1000 4000 4000 4000];
        else
                info = [1000 1000 1000 4000 4000 4000].* w_loop;
        end
        w = sparse(infoi,infoj,info,6,6);
        Ti = T{i};
        Tj = T{j}; 
        Zij = edge_store{iter2,2};

        TjinTi = Tj\Ti; % 右乘
        eij = SE32se3_back(Zij*((Ti)\(Tj))); % 6-by-1
        Jr = eye(6)+0.5.*...
            [0     -eij(6)  eij(5) 0     -eij(3) eij(2);
            eij(6)  0      -eij(4) eij(3) 0     -eij(1);
            -eij(5)  eij(4) 0     -eij(2) eij(1) 0     ;
            0       0       0      0     -eij(6) eij(5);
            0       0       0      eij(6) 0     -eij(4);
            0       0       0     -eij(5) eij(4) 0 ];
    
        AdTj = [TjinTi(1:3,1:3) ...
            ([0 -TjinTi(15) TjinTi(14);TjinTi(15) 0 -TjinTi(13);-TjinTi(14) TjinTi(13) 0]*TjinTi(1:3,1:3));...
            zeros(3) TjinTi(1:3,1:3)];
        Aij = -Jr*AdTj ; % 6-by-6
        Bij =  Jr ; % 6-by-6
        ia = (i*6)-5; ib = i*6; ja = (j*6)-5; jb = j*6;

        H_ij = Aij' * w * Bij;% 6-by-6
    
        H(ia:ib,ia:ib) = H(ia:ib,ia:ib) + Aij' * w * Aij;
        H(ia:ib,ja:jb) = H(ia:ib,ja:jb) + H_ij;
        H(ja:jb,ia:ib) = H(ja:jb,ia:ib) + H_ij';
        H(ja:jb,ja:jb) = H(ja:jb,ja:jb) + Bij' * w * Bij;
        b(ia:ib) = b(ia:ib) + Aij' * w * eij;
        b(ja:jb) = b(ja:jb) + Bij' * w * eij;
    end
    H = H(7:end,7:end); 
    b = b(7:end);
    % GN
%     delta_x_lie = H\(-b);
%     delta_x_lie = chol(H)\(chol(H)'\(-b));

    % LM
    delta_x_lie = (H+mu*speye(pose_num*6-6))\(-b); 

    delta_x_lie = [0 0 0 0 0 0;(reshape(delta_x_lie,6,pose_num-1))']; % lie algebra
    delta_rms = rms(delta_x_lie);
    delta_T = cell(pose_num,1);
    for i = 1:pose_num
       lie = delta_x_lie(i,:); 
       Tn = se32SE3(lie);
       delta_T{i} = Tn;
    end
    
    T = cellfun(@mtimes,T,delta_T,'UniformOutput',false); % 右乘
  
      % threshold
    if (delta_rms < threshold) 
        break
    end
    % LM mu updating
    if condest(H)>1000 
        mu = mu*10 ;
    else 
        mu = mu/10; 
    end
    
end
Poses = T;