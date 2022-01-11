function getDataTrain(brainNo, fileDir, cellType, fid1, fid2, annotDir, jpegDir, tileSize, bitType)



% for i = 1 : length(brains)
%     brainNo = brains{i};
    disp(brainNo);
    if cellType == 'R'
        chnl = 1;
    end
    if cellType == 'G'
        chnl = 2;
    end
    
    imgDir = [fileDir '/annotated' cellType '/' brainNo '/images/'];
    
    fileList = dir(fullfile([fileDir 'annotated' cellType '/' brainNo], ...
        '*.json'));
    
    for i = 1 : length(fileList)
        disp(fileList(i).name);
        fileName1 = strrep(fileList(i).name, '&', '__');
        atlasJSON = loadjson([fileDir 'annotated' cellType '/' brainNo ...
            '/' fileList(i).name]);
        if exist([imgDir fileList(i).name(1:end-4) 'jp2'])
            img = imread([imgDir fileList(i).name(1:end-4) 'jp2']);
        else
            continue;
        end
        mask = generateCellMask(atlasJSON, img);
        
        for row = 1 : tileSize : size(img,1) - tileSize -1
            for col = 1 : tileSize : size(img,2) - tileSize -1
                %                 imgTile = uint8(zeros(1024,1024,3));
                imgTile = uint8(floor(img(row:row+tileSize-1, col:col+tileSize-1, chnl)/bitType));
                %                 imgTile = (imgTile - min(min(imgTile)))/(max(max(imgTile)) - min(min(imgTile))) * 255;
                maskTile = mask(row:row+tileSize-1, col:col+tileSize-1);
                
                BW_pad = false(tileSize,tileSize);
                BW = imbinarize(maskTile);
                BW_pad(1:size(BW,1), 1:size(BW,2)) = BW;
                numCenters = sum(sum(BW_pad));
                
                if numCenters
                    
                    img_pad = uint8(zeros(tileSize,tileSize,3));
                    
                    % Todo: Change the data range for uint16 images
                    img_pad(1:size(imgTile,1), 1:size(imgTile,2),chnl) = imgTile;
                    
                    imwrite(img_pad, [jpegDir fileName1(1:end-5) ...
                        '_' num2str(row) '_' num2str(col) '_' cellType '.jpg']);
                    
                    fprintf(fid1, '%s\n', [fileName1(1:end-5) '_' ...
                        num2str(row) '_' num2str(col) '_' cellType]);
                    fprintf(fid2, '%s\n', [fileName1(1:end-5) '_' ...
                        num2str(row) '_' num2str(col) '_' cellType]);
                    
                    filename = [annotDir fileName1(1:end-5) ...
                        '_' num2str(row) '_' num2str(col) '_' cellType '.xml'];
                    
                    fid = fopen(filename, 'w');
                    
                    fprintf(fid, '<annotation>\n\t<folder>VOC2007</folder>\n\t<filename>%s</filename>', ...
                        [fileName1(1:end-5) '_' num2str(row) '_' ...
                        num2str(col) '_' cellType '.jpg']);
                    fprintf(fid, '\n\t<size>\n\t\t<width>1024</width>\n\t\t<height>1024</height>');
                    fprintf(fid, '\n\t\t<depth>1</depth>\n\t</size>\n\t<segmented>0</segmented>');
                    
                    [x, y] = find(BW_pad);
                    for j = 1  : numCenters
                        maxX = min(x(j) + 10, size(BW_pad,1));
                        maxY = min(y(j) + 10, size(BW_pad,2));
                        minX = max(x(j) - 10, 1);
                        minY = max(y(j) - 10, 1);
                        
                        fprintf(fid, '\n\t<object>');
                        fprintf(fid, '\n\t\t<name>');
%                         fprintf(fid, cellType);
                        fprintf(fid, 'cell</name>\n\t\t<pose>Frontal</pose>');
                        fprintf(fid, '\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>');
                        fprintf(fid, '\n\t\t<bndbox>');
                        fprintf(fid, '\n\t\t\t<xmin>%d</xmin>', minY);
                        fprintf(fid, '\n\t\t\t<ymin>%d</ymin>', minX);
                        fprintf(fid, '\n\t\t\t<xmax>%d</xmax>', maxY);
                        fprintf(fid, '\n\t\t\t<ymax>%d</ymax>', maxX);
                        fprintf(fid, '\n\t\t</bndbox>\n\t</object>');
                    end
                    
                    fprintf(fid, '\n</annotation>');
                    fclose(fid);
                end
            end
        end
        
    end
end

%     fprintf(fidTxtDone, '%s\n', brainNo);
%     brainNo = fgetl(fidTxtReady);
% end


% fclose(fidTxtReady);
% fclose(fidTxtDone);
% end
