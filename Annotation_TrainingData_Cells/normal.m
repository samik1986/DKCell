function [norm_img]=normal(img)
    img=double(img);
    norm_img=img-min(img(:));
    norm_img=norm_img./(max(img(:))- min(img(:)));
    norm_img=im2uint16(norm_img);
    %norm_img(norm_img<3500)=0;
%     norm_img(norm_img<1500)=0;
end