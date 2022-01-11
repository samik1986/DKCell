rootDir = '/home/samik/mnt/gpu5b_1/nfs/data/main/M32/Training_Data/Keras-FasterRCNN/dataF/VOCdevkit/VOCDevkit2007/VOC2007/';

img1Dir = [rootDir 'JPEGImages/'];
xmlDir = [rootDir 'Annotations/'];
opDir = [rootDir 'AnnotVerify/'];
mkdir(opDir);
xmlFiles = dir(fullfile(xmlDir, '*.xml'));

for count = 1 : length(xmlFiles)
    [~,tile1] = fileparts(xmlFiles(count).name);
    % tile1 = 'PMD1024__1023-F13-2012.06.04-15.45.34_PMD1024_2_0038_11265_13313';
    annot1 = parseXML([xmlDir tile1 '.xml']);
    
    img1 = imread([img1Dir tile1 '.jpg']);
    
    f = figure; 
    imshow(img1*8,'Border', 'tight'); hold on;
    
    for ii = 1  : length(annot1.Children)
        if strcmp(annot1.Children(ii).Name, 'object')
            for jj = 1 : length(annot1.Children(ii).Children)
                if strcmp(annot1.Children(ii).Children(jj).Name, 'bndbox')
                    for kk = 1 : length(annot1.Children(ii).Children(jj).Children)
                        if strcmp(annot1.Children(ii).Children(jj).Children(kk).Name, 'xmin')
                            xmin = str2num(annot1.Children(ii).Children(jj).Children(kk).Children.Data);
                        end
                        if strcmp(annot1.Children(ii).Children(jj).Children(kk).Name, 'ymin')
                            ymin = str2num(annot1.Children(ii).Children(jj).Children(kk).Children.Data);
                        end
                        if strcmp(annot1.Children(ii).Children(jj).Children(kk).Name, 'xmax')
                            xmax = str2num(annot1.Children(ii).Children(jj).Children(kk).Children.Data);
                        end
                        if strcmp(annot1.Children(ii).Children(jj).Children(kk).Name, 'ymax')
                            ymax = str2num(annot1.Children(ii).Children(jj).Children(kk).Children.Data);
                        end
                    end
                    if strcmp(annot1.Children(ii).Children(2).Children.Data,'red')
                        disp('Red')
                        rectangle('Position', [xmin ymin xmax-xmin ymax-ymin],'EdgeColor','c');
                        text(xmax-15,ymin+15, 'R', 'Color','c');
                    end
                    if strcmp(annot1.Children(ii).Children(2).Children.Data,'green')
                        disp('Green')
                        rectangle('Position', [xmin ymin xmax-xmin ymax-ymin],'EdgeColor','m');
                        text(xmin,ymin+15, 'G', 'Color','y');
                    end
                    if strcmp(annot1.Children(ii).Children(2).Children.Data,'Fast Blue')
                        disp('FB')
                        rectangle('Position', [xmin ymin xmax-xmin ymax-ymin],'EdgeColor','m');
%                         text(xmin,ymin+15, 'FB', 'Color','y');
                    end
                    %                 disp(xmin); disp(ymin); disp(xmax-xmin); disp(ymax-ymin);
                end
            end
        end
        
    end
    print(f, '-r80', '-dtiff', [opDir tile1 '.tif']);
    hold off;
    close all;
end
    % xmin = 37;
    % xmax = 57;
    % ymin = 988;
    % ymax = 1008;
    % rectangle('Position', [xmin ymin xmax-xmin ymax-ymin],'EdgeColor','w')