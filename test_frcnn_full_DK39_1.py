from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import tensorflow as tf
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from skimage.io import imread, imsave
# from skimage.transform import rotate
from scipy import misc 
from scipy.io import loadmat
from keras_frcnn import config, data_generators
import h5py
import hdf5storage
import bisect
import statistics
from scipy.ndimage.morphology import binary_fill_holes
import fnmatch
from scipy.ndimage.interpolation import rotate
from getRegMask_DK39 import getRegMask
import json
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth=True
sess = tf.Session(config=cfg)
sys.setrecursionlimit(40000)

parser = OptionParser()
parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=50)
parser.add_option("--config_filename", dest="config_filename", help="Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
(options, args) = parser.parse_args()

# config_output_filename = options.config_filename
# with open(config_output_filename, 'rb') as f_in:
#     C = pickle.load(f_in)

C = config.Config()

print(C)

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'xception':
    import keras_frcnn.xception as nn
elif C.network == 'inception_resnet_v2':
    import keras_frcnn.inception_resnet_v2 as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
img_path = options.test_path


model_path = '/nfs/data/main/M32/Training_Data/Keras-FasterRCNN/model_frcnn_DK_May_2021.hdf5'

################################################

brainNo = sys.argv[1]
flagReg = sys.argv[2]
listmin = int(sys.argv[3])
listmax = int(sys.argv[4])
resolution = float(sys.argv[5])
redo = int(sys.argv[6])

print("0. " + brainNo + " " + flagReg + " " + str(listmin) + " " + str(listmax) + " " + str(resolution) + " " + str(redo))
# resolution = float(sys.argv[7])

# models_albu = albu_dingkang.read_model([os.path.join('/nfs/data/main/M32/Training_Data/ProcessDet/weights/MBA_Jul2021/', 'fold{}_best.pth'.format(i)) for i in range(4)])

filePath = '/home/samik/Keras-FasterRCNN/images/'
if flagReg == '1':
    filePath = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/' + brainNo + '_img/'
    maskDir = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/' + brainNo + '_json/' #reg_high_seg_pad/'
    outDir = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_reg/' + brainNo + '/'
else:
    outDir = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_unreg/' + brainNo + '/'

# filePath = 'temp/'
# fileList1 = os.listdir(filePath)

# DK39 adhoc:
filePath = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/' + brainNo + '_img/'
maskDir = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/dk39_output_v07/2D_neurotrace_space/'
#########################

f=open(outDir + 'listF.txt')
fileList1=f.readlines()[listmin: listmax]
f.close()
# if ~legacy:
#     maskDir = '/nfs/data/main/M32/RegistrationData/Data_ Marmoset/' + brainNo + '/Transformation_OUTPUT/reg_high_seg_pad/'
outDir = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_reg/' + brainNo + '/'
os.system("mkdir " + outDir)
outDirL = outDir + 'mask/'
os.system("mkdir " + outDirL)
# outDirG = '/nfs/data/main/M32/Process_Detection/ProcessDetPass1/' + brainNo + '/'
# os.system("mkdir " + outDir)
if redo:
    fileList2 = [] #os.listdir(outDir)
else:
    fileList2 = os.listdir(outDir)

###################################################
# brainNo = 'm826F'
# filePath = '/home/samik/Keras-FasterRCNN/images/'
# # filePath = '/nfs/data/main/M31/KleinfeldU19/training_images/DK55/CH3/cell_detectionJP2/Reqd/'
# # filePath = 'tmp/'

# # outDirR = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_reg/' + brainNo + '/'
# outDirR = 'tmpOut/'
# outDirR = ''
# jsonOutG = os.path.join(outDirR, 'jsonG/')
# jsonOutF = os.path.join(outDirR, '/nfs/data/main/M31/KleinfeldU19/training_images/DK55/CH3/cell_detectionJSON/')
# maskOut = os.path.join(outDirR, '/nfs/data/main/M31/KleinfeldU19/training_images/DK55/CH3/cell_detectionOP/')
jsonOutF = os.path.join(outDir, 'jsonF/')
# maskOut = os.path.join(outDirR, 'mask/')


# os.system("mkdir " + outDirR)
# os.system("mkdir " + maskOut)
os.system("mkdir " + jsonOutF)
# os.system("mkdir " + jsonOutG)

# fileList1 = os.listdir(filePath)
# # print(fileList1)
# fileList2 = [] #os.listdir(maskOut)

# for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
#     if not(fnmatch.fnmatch(fichier, '*.jp2')):
#         fileList1.remove(fichier)

# for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
#     if not(fnmatch.fnmatch(fichier, '*F*')):
#         fileList1.remove(fichier)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
        if fichier in fileList2[:]:
            fileList1.remove(fichier)

def imread_fast(img_path):
    img_path = img_path[0:-1]
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    err_code = os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16")
    img = imread('temp/'+base+'.tif')
    os.system("rm temp/"+base_C+'.tif')
    return img

def imwrite_fast(img_path, opImg):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    img = imsave('temp1/'+base+'.tif', opImg)
    err_code = os.system("kdu_compress -i temp1/"+base_C+".tif -o "+img_path_C+" -rate 1 Creversible=yes Clevels=7 Clayers=8 Stiles=\{1024,1024\} Corder=RPCL Cuse_sop=yes ORGgen_plt=yes ORGtparts=R Cblk=\{32,32\} -num_threads 32")
    os.system("rm temp1/"+base_C+'.tif')

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width ,_) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img_channels(img, C):
    """ formats the image channels based on config """
    # img = img[:, :, (2, 1, 0)] #BGR->RGB
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'xception':
    num_features = 1024
elif C.network == 'inception_resnet_v2':
    num_features = 1024
elif C.network == 'vgg':
    num_features = 512

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(model_path))
model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.5

visualise = True

for files in fileList1:
    print(os.path.join(filePath, files))
    image1 = imread_fast(os.path.join(filePath, files))
    print(image1.shape)
    image1 = image1//16
    image1 = np.clip(image1, 0, 255)
    image1 = image1.astype(np.uint8) 
    # image1 = np.rot90(image1, -1)
    # imwrite_fast(os.path.join(maskOut,files), image1)
    org_w, org_h, org_c = image1.shape
    image = np.zeros((org_w,org_h,3),dtype=np.uint8)
    image[:, :, 1] = image1[:,:,2]
    # imwrite_fast(os.path.join(outDirL, files.replace('\n', '')), image)
    del image1
    if os.path.exists(os.path.join(maskDir,files.replace('jp2\n', 'mat'))):
                    maskB = hdf5storage.loadmat(os.path.join(maskDir,files.replace('jp2\n', 'mat')))['seg']
    elif os.path.exists(os.path.join(maskDir, 'atlas-seg_to_' + files.replace('.jp2\n', '_small.geojson'))):
        with open(os.path.join(maskDir, 'atlas-seg_to_' + files.replace('.jp2\n', '_small.geojson'))) as fAtlas:
            jAtlas = json.load(fAtlas)
        maskB = getRegMask(jAtlas, org_w, org_h, resolution)
    else: 
        print("No Tissue mask!!!") 
        maskB = np.ones((org_w,org_h),dtype='uint8')
    
    # maskB = hdf5storage.loadmat(os.path.join(maskDir,f1.replace('jp2', 'mat')))['seg']
    maskB = maskB / maskB.max()
    # print(maskB.shape)
    _,maskB = cv2.threshold(maskB,0,1,cv2.THRESH_BINARY)
    maskB = np.uint8(maskB) * 255
    # imwrite_fast(os.path.join(outDirL, files.replace('.jp2\n', '_mask.jp2')), maskB)
    # image = np.zeros((org_w,org_h,3),dtype=np.uint8) 
    # image[:, :, 0] = 0
    # image[:, :, 1] = image1
    # image[:, :, 2] = 0
    # imwrite_fast(os.path.join(maskOut,files), image))
    # op = np.zeros((org_w, org_h), dtype='uint8')
    # op1 = np.zeros((org_w,org_h,3), dtype='uint8')
    # print(org_w,org_h)
    window = 1024
    count = 0
    st = time.time()
    ##################################################################################

    f = open(os.path.join(jsonOutF, files.replace('jp2\n', 'json')), "w")
    f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"0\",\"properties\":{\"name\":\"Cells\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":[")
    
    f1 = open(os.path.join(jsonOutF, 'check_det.csv'), "w")

    for row in range(0, org_w - window, window):
        for col in range(0, org_h - window, window):
            img = np.zeros((window, window, 3), dtype= 'uint8')
            img[:, :, 0] = image[row:row + window, col:col + window, 1]
            tileM = maskB[row:row + window, col:col + window]
            tileOP = np.zeros((window, window), dtype='uint8')
            # img = np.zeros((window, window, 3), dtype= 'uint8')
            if np.sum(tileM):
                # img = tile #
                # img1  = image[row:row + window, col:col + window, :]
                X, ratio = format_img(img, C)
                X = np.transpose(X, (0, 2, 3, 1))
                # get the feature maps and output from the RPN
                [Y1, Y2, F] = model_rpn.predict(X)
                R = roi_helpers.rpn_to_roi(Y1, Y2, C, overlap_thresh=0.7)
                # convert from (x1,y1,x2,y2) to (x,y,w,h)
                R[:, 2] -= R[:, 0]
                R[:, 3] -= R[:, 1]
                # apply the spatial pyramid pooling to the proposed regions
                bboxes = {}
                probs = {}
                for jk in range(R.shape[0]//C.num_rois + 1):
                    ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                    if ROIs.shape[1] == 0:
                        break
                    if jk == R.shape[0]//C.num_rois:
                        #pad R
                        curr_shape = ROIs.shape
                        target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                        ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                        ROIs_padded[:, :curr_shape[1], :] = ROIs
                        ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                        ROIs = ROIs_padded
                    [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
                    for ii in range(P_cls.shape[1]):
                        if np.max(P_cls[0, ii, :]) < bbox_threshold: # or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                            continue
                        cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                        if cls_name not in bboxes:
                            bboxes[cls_name] = []
                            probs[cls_name] = []
                        (x, y, w, h) = ROIs[0, ii, :]
                        cls_num = np.argmax(P_cls[0, ii, :])
                        try:
                            (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                            tx /= C.classifier_regr_std[0]
                            ty /= C.classifier_regr_std[1]
                            tw /= C.classifier_regr_std[2]
                            th /= C.classifier_regr_std[3]
                            x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                        except:
                            pass
                        bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                        probs[cls_name].append(np.max(P_cls[0, ii, :]))
                all_dets = []
                # print(bboxes)
                
                for key in bboxes:
                    if key == 'cell':
                        bbox = np.array(bboxes[key])
                        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
                        for jk in range(new_boxes.shape[0]):
                            (x1, y1, x2, y2) = new_boxes[jk,:]
                            # print(new_boxes[jk,:])
                            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                            
                            all_dets.append((key,100*new_probs[jk]))
                            cen_x = (real_x1 + real_x2)/2
                            cen_y = (real_y1 + real_y2)/2
                            # print(cen_x, cen_y)
                            f1.write(str(org_w) + ',' + str(row+np.int64(cen_y)) + "," + str(col + np.int64(cen_x)) + "\n" )
                            f.write("[" + str(col + np.int64(cen_x)) + "," + str((row + np.int64(cen_y))*-1) + "],")
                            # f.write("[" + str(np.uint16(cen_y + row)) + "," + str(np.uint16(cen_x + col) * -1) + "],")

                            # print(np.uint16(cen_y + row), np.uint16(cen_x + col))
                            # cv2.rectangle(img1, (real_x1, real_y1), (real_x2, real_y2), (
                            # int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)
                            
                            # textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                            # all_dets.append((key, 100 * new_probs[jk]))
                            
                            # (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                            # textOrg = (real_x1, real_y1 - 0)
                            
                            # cv2.rectangle(img1, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                            #               (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                            # cv2.rectangle(img1, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                            #               (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                            # cv2.putText(img1, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
                #             tileOP[np.uint16(cen_x), np.uint16(cen_y)] = 255
                # op[row:row + window, col:col + window] = tileOP
                # op1[row:row + window, col:col + window,:] = img1

    # _, thresh = cv2.threshold(op, 127, 255, cv2.THRESH_BINARY)
    # _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
    # f = open(os.path.join(jsonOutF, files.replace('jp2', 'json')), "w")
    # f.write(
    #     "{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"0\",\"properties\":{\"name\":\"PreMotor Cells\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":[")
    # for pts in centroids:
    #     pts64 = pts.astype(np.int64)
    #     f.write("[" + str(pts64[0]) + "," + str(pts64[1] * -1) + "],")
    f.write("[]]}}]}")
    f.close()
    f1.close()
    # cv2.imwrite(os.path.join(maskOut,files), op1)
    # imsave(os.path.join(maskOut,files), image1)

    print('Elapsed time = {}'.format(time.time() - st))

## Cerate Version History ##
f1=open(outDir + 'version_history.txt', 'w')
now = datetime.now()
f1.write(str(now))
f1.write('\n')
if flagReg == '1' :
    f1.write('Registered : Y\n')
else:
    f1.write('Registered : N\n')

f1.write('Input Path:' + filePath + '\n')
f1.write('Output Path:' + outDir + '\n')
fileList2 = os.listdir(jsonOutF)
f1.write('Number of Files Detected:' + str(len(fileList2))+ '\n')

f1.close()
