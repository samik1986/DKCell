from __future__ import print_function
import numpy as np
import cv2
from skimage.io import imread, imsave
import os
# from scipy.ndimage.io import imread

def imread_fast(img_path):
    # img_path = img_path[0:-1]
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    err_code = os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16")
    img = imread('temp/'+base+'.tif')
    os.system("rm temp/"+base_C+'.tif')
    return img

# In[3]:
def imwrite_fast(img_path, opImg):
    # img_path = img_path[0:-1]
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    img = imsave('temp1/'+base+'.tif', opImg) # Needs a temp folder for intermediate TIFF image in the CWD
    err_code = os.system("kdu_compress -i temp1/"+base_C+".tif -o "+img_path_C+" -rate 1 Creversible=yes Clevels=7 Clayers=8 Stiles=\{1024,1024\} Corder=RPCL Cuse_sop=yes ORGgen_plt=yes ORGtparts=R Cblk=\{32,32\} -num_threads 32")
    os.system("rm temp1/"+base_C+'.tif')


bg_image_path = '/nfs/data/main/M32/RegistrationData/Data/DK39/DK39_img/DK39_ID_0607_slide053_S3_C1.jp2'
fg_image_path = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_Nissl/Annotation_tool/DK39/DK39_ID_0607_slide053_S3_C1.jp2'
# Opening the primary image (used in background)
frontImage = imread_fast(fg_image_path)
# Opening the secondary image (overlay image)
background = imread_fast(bg_image_path)
alpha = 0.5
# create two copies of the original image -- one for
# the overlay and one for the final output image
overlay = frontImage.copy()
output = background.copy()
cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
imwrite_fast('new.jp2', output)
# # Convert image to RGBA
# frontImage = frontImage.convert("RGBA")
  
# # Convert image to RGBA
# background = background.convert("RGBA")
  
# # Calculate width to be at the center
# width = (background.width - frontImage.width) // 2
  
# # Calculate height to be at the center
# height = (background.height - frontImage.height) // 2
  
# # Paste the frontImage at (width, height)
# background.paste(frontImage, (width, height), frontImage) 
# # Save this image
# background.save("new.jp2", format="jp2")
