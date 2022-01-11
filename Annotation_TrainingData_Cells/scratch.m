imgDir = '/home/samik/mnt/gpu5a/Keras-FasterRCNN/data/VOCdevkit/VOCDevkit2007/VOC2007/JPEGImages/';
xmlDir = '/home/samik/mnt/gpu5a/Keras-FasterRCNN/data/VOCdevkit/VOCDevkit2007/VOC2007/Annotations/';
tile = 'M6344-F34--_2_0068_7169_17409';
img = imread([imgDir tile '.jpg']);
annot = parseXML([xmlDir tile '.xml']);

figure; imshow(img); hold on;

for i = 1  : length(annot.Children)
    if strcmp(annot.Children(i).Name, 'object')
        for j = 1 : length(annot.Children(i).Children)
            if strcmp(annot.Children(i).Children(j).Name, 'bndbox')
                for k = 1 : length(annot.Children(i).Children(j).Children)
                    if strcmp(annot.Children(i).Children(j).Children(k).Name, 'xmin')
                        xmin = str2num(annot.Children(i).Children(j).Children(k).Children.Data);
                    end
                    if strcmp(annot.Children(i).Children(j).Children(k).Name, 'ymin')
                        ymin = str2num(annot.Children(i).Children(j).Children(k).Children.Data);
                    end
                    if strcmp(annot.Children(i).Children(j).Children(k).Name, 'xmax')
                        xmax = str2num(annot.Children(i).Children(j).Children(k).Children.Data);
                    end
                    if strcmp(annot.Children(i).Children(j).Children(k).Name, 'ymax')
                        ymax = str2num(annot.Children(i).Children(j).Children(k).Children.Data);
                    end
                end
                
                rectangle('Position', [xmin ymin xmax-xmin ymax-ymin],'EdgeColor','r')
                
            end
        end
    end
                    
end



%%
% for j = 1  : numCenters    
%     minX = max(x(j) - 15, 1);
%     
%     minY = max(y(j) - 15, 1);
%     disp(minY);
%     rectangle('Position', [minY minX 10 10],'EdgeColor','r')
% end
