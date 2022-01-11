direct = '/home/samik/mnt/gpu5b_1/nfs/data/main/M32/Training_Data/Keras-FasterRCNN/dataF/VOCdevkit/VOCDevkit2007/VOC2007/cellImages/';

direc = dir(fullfile(direct, '*.jpg'));

% fh = figure();
% fh.WindowState = 'maximized';
% hold on;

h = [];
for i = 1 :  length(direc)
    img = imread(fullfile(direct, direc(i).name));
    h1 = imhist(img(:,:,3));
%     h = [h; h1(1:50)'];
    
    
%     h1 = 
    
%     plot(h);
%     fh.WindowState = 'maximized';
    subplot(1,2,1);imshow(img*8);
    subplot(1,2,2);plot(h1(1:50));
%     if ~mod(i,20)
        pause;
%     end

end

hMean = mean(h);

% Y = tsne(h,'Algorithm','barneshut','NumPCAComponents',50);
% 
% figure
% gscatter(Y(:,1),Y(:,2),L)