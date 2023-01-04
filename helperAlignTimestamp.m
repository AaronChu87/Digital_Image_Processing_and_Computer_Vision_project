function indexPairs = helperAlignTimestamp(timeColor, timeDepth)
idxDepth = 1;
indexPairs = zeros(numel(timeColor), 2);
for i = 1:numel(timeColor)
    for j = idxDepth : numel(timeDepth)
        if abs(timeColor(i) - timeDepth(j)) < 1e-4
            idxDepth = j;
            indexPairs(i, :) = [i, j];
            break
        elseif timeDepth(j) - timeColor(i) > 1e-3
            break
        end
    end
end
indexPairs = indexPairs(indexPairs(:,1)>0, :);
end