%% Matlab Code to generate Training data for the Faster RCNN from manual point annotations given as JSON files.
% ------Samik Banerjee (October 5, 2020).
% Command line arguments for running
% Needs JSONLab library
% matlab -nodisplay -nosplash -nodesktop -r "run('genCellTrainData.m'); exit();"
% Input1 - txt file (filenamesReady.txt) for the annotated brain nos. (line 17)
% Input2 - Original image files directory corresponding to the annotation (line 18)
% Outputs - $Keras-FasterRCNN$/data/VOCdevkit/VOCDevkit2007/VOC2007/ (line 27 - 29)

%% Initialisations

% function genCellTrainData()
addpath(genpath('jsonlab'));



fileDir = '/nfs/data/main/M32/AnnotatedResults/syn_results/wholebraincells/';
cellType = 'G'; % F = FastBlue, R = Red Tracer, G = GreenTracer





imgDir = '/nfs/data/main/M25/marmosetRIKEN/NZ/';

% cellType = 'G'; % F = FastBlue, R = Red Tracer, G = GreenTracer
channelType = 2; % 3 = FastBlue, 1 = Red Tracer, 2 = GreenTracer
bitType = 16; % 1 for 8-bit data ; use 16 (2^4) for PMD (12-bit) and 256 (2^8) for other 16-bit data

%% Brains ready for Retraining
% fidTxtReady = fopen([fileDir 'annotated' cellType '/' 'filenamesReady.txt']);
% fidTxtDone = fopen([fileDir 'annotated' cellType '/' 'filenamesDone.txt'], 'a');

%% Create Faster RCNN compatible training set
annotDir = '/home/samik/Keras-FasterRCNN/data/VOCdevkit/VOCDevkit2007/VOC2007/Annotations/';
imageSetsDir = '/home/samik/Keras-FasterRCNN/data/VOCdevkit/VOCDevkit2007/VOC2007/ImageSets/Main/';
jpegDir = '/home/samik/Keras-FasterRCNN/data/VOCdevkit/VOCDevkit2007/VOC2007/JPEGImages/';

tileSize = 1024;

system(['rm ' annotDir '*']);
system(['rm ' jpegDir '*']);

fid1 = fopen([imageSetsDir 'trainval.txt'], 'w');
fid2 = fopen([imageSetsDir 'train.txt'], 'w');

brains = {'m6344', 'm6328'}; % Name the brains for training
for i = 1 : length(brains)
    brainNo = brains{i};
    disp(brainNo);
    
    fileList = dir(fullfile([fileDir 'annotated' cellType '/' brainNo], ...
        '*.json'));
    
    parfor i = 1 : length(fileList)
        disp(fileList(i).name);
        atlasJSON = loadjson([fileDir 'annotated' cellType '/' brainNo ...
            '/' fileList(i).name]);
        img = imread([imgDir brainNo '/' brainNo 'F/JP2/' ...
            '/' fileList(i).name(1:end-4) 'jp2']);
        mask = generateCellMask(atlasJSON, img);
        
        for row = 1 : tileSize : size(img,1) - tileSize -1
            for col = 1 : tileSize : size(img,2) - tileSize -1
                %                 imgTile = uint8(zeros(1024,1024,3));
                imgTile = uint8(floor(img(row:row+tileSize-1, col:col+tileSize-1, :)/bitType));
                %                 imgTile = (imgTile - min(min(imgTile)))/(max(max(imgTile)) - min(min(imgTile))) * 255;
                maskTile = mask(row:row+tileSize-1, col:col+tileSize-1);
                
                BW_pad = false(tileSize,tileSize);
                BW = imbinarize(maskTile);
                BW_pad(1:size(BW,1), 1:size(BW,2)) = BW;
                numCenters = sum(sum(BW_pad));
                
                if numCenters
                    
                    img_pad = uint8(zeros(tileSize,tileSize,3));
                    
                    % Todo: Change the data range for uint16 images
                    img_pad(1:size(imgTile,1), 1:size(imgTile,2),:) = imgTile;
                    
                    imwrite(img_pad, [jpegDir fileList(i).name(1:end-5) ...
                        '_' num2str(row) '_' num2str(col) '.jpg']);
                    fprintf(fid1, '%s\n', [fileList(i).name(1:end-5) '_' ...
                        num2str(row) '_' num2str(col)]);
                    fprintf(fid2, '%s\n', [fileList(i).name(1:end-5) '_' ...
                        num2str(row) '_' num2str(col)]);
                    
                    filename = [annotDir fileList(i).name(1:end-5) ...
                        '_' num2str(row) '_' num2str(col) '.xml'];
                    fid = fopen(filename, 'w');
                    fprintf(fid, '<annotation>\n\t<folder>VOC2007</folder>\n\t<filename>%s</filename>', ...
                        [fileList(i).name(1:end-5) '_' num2str(row) '_' ...
                        num2str(col) '.jpg']);
                    fprintf(fid, '\n\t<size>\n\t\t<width>1024</width>\n\t\t<height>1024</height>');
                    fprintf(fid, '\n\t\t<depth>1</depth>\n\t</size>\n\t<segmented>0</segmented>');
                    
                    [x, y] = find(BW_pad);
                    for j = 1  : numCenters
                        maxX = min(x(j) + 10, size(BW_pad,1));
                        maxY = min(y(j) + 10, size(BW_pad,2));
                        minX = max(x(j) - 10, 1);
                        minY = max(y(j) - 10, 1);

                        fprintf(fid, '\n\t<object>\n\t\t<name>cell</name>\n\t\t<pose>Frontal</pose>');
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

fclose(fid1);
fclose(fid2);
% fclose(fidTxtReady);
% fclose(fidTxtDone);
% end
