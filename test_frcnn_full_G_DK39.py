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
from skimage.io import imread
from scipy import misc 
from scipy.io import loadmat
import h5py
import hdf5storage
import bisect
import statistics
from scipy.ndimage.morphology import binary_fill_holes
import fnmatch


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth=True
sess = tf.Session(config=cfg)

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help="Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

# if not options.test_path:   # if filename is not given
#     parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

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

model_path = 'model_frcnn.hdf5'
brainNo = 'm919F'

# filePath = '/nfs/data/main/M32/Samik/test/'
filePath = '/nfs/data/main/M25/marmosetRIKEN/NZ/m919/m919F/JP2/'
# filePath = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/' + brainNo + '_img/' #reg_high_tif_pad_jp2/'
# maskDir = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/reg_high_seg_pad/'

outDirR = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_reg/' + brainNo + '/'



filePath = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/' + brainNo + '_img/' #reg_high_tif_pad_jp2/'
filePath = 'tmp1/'
maskDir = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/reg_high_seg_pad/'
maskDir = 'tmp1/'
outDirR = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_reg/' + brainNo + '/'
outDirR = 'tmpOut/'

jsonOutG = os.path.join(outDirR, 'jsonG/')
jsonOutR = os.path.join(outDirR, 'jsonR/')
maskOut = os.path.join(outDirR, 'mask/')


os.system("mkdir " + outDirR)
os.system("mkdir " + maskOut)
os.system("mkdir " + jsonOutR)
os.system("mkdir " + jsonOutG)

fileList1 = os.listdir(filePath)
print(fileList1)
fileList2 = [] #os.listdir(maskOut)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*.jp2')):
        fileList1.remove(fichier)

# for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
#     if not(fnmatch.fnmatch(fichier, '*F*')):
#         fileList1.remove(fichier)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
        if fichier in fileList2[:]:
            fileList1.remove(fichier)

def imread_fast(img_path):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    err_code = os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16")
    img = imread('temp/'+base+'.tif')
    os.system("rm temp/"+base_C+'.tif')
    return img

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
    img = img[:, :, (2, 1, 0)]
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

bbox_threshold = 0.1

visualise = True

for files in fileList1:
    print(files)
    image1 = imread_fast(
        os.path.join('/nfs/data/main/M32/RegistrationData/Data/DK39/Transformation_OUTPUT_CH2/DK39_img/', files))
    w, h = image1.shape
    image = np.zeros((w, h, 3), dtype='uint16')
    image[:, :, 0] = image1
    image[:, :, 1] = imread_fast(
        os.path.join('/nfs/data/main/M32/RegistrationData/Data/DK39/Transformation_OUTPUT_CH3/DK39_img/', files))
    image[:, :, 2] = imread_fast(
        os.path.join('/nfs/data/main/M32/RegistrationData/Data/DK39/Transformation_OUTPUT_CH1/DK39_img/', files))
    print(image.shape)
    # maskB = hdf5storage.loadmat(os.path.join(maskDir,files.replace('jp2', 'mat')))['seg']
    maskB = imread(os.path.join(maskDir, files.replace('jp2', 'tif')))
    maskB = maskB / maskB.max()

    ret, maskB = cv2.threshold(maskB, 0, 1, cv2.THRESH_BINARY)
    maskB = np.uint8(maskB) * 255

    image = image // 256
    image = image.astype(np.uint8)
    op = np.zeros((w,h),dtype='uint8')
    op1 = np.zeros((w, h,3), dtype='uint8')
    # opR = np.zeros((w,h),dtype='bool')
    # opG = np.zeros((w,h),dtype='bool')
    print(w,h)
    window = 1024
    count = 0
    st = time.time()
    for row in range(0, w - window, window):
        for col in range(0, h - window, window):
            print(row, col)
            # print(row+window, col+window)
            # maskR = np.zeros((window, window), dtype='bool')
            # maskG = np.zeros((window, window), dtype='bool')
            # mask = np.zeros((window, window), dtype='uint8')
            tile = image[row:row + window, col:col + window, :]
            tileM = maskB[row:row + window, col:col + window]
            tileOP = np.zeros((window,window), dtype='uint8')
            img = np.zeros((window, window,3), dtype='uint8')
            if np.sum(tileM):
                img = tile
                # print("yes...............")
                X, ratio = format_img(img, C)
                # if K.image_dim_ordering() == 'tf':
                X = np.transpose(X, (0, 2, 3, 1))
                # get the feature maps and output from the RPN
                [Y1, Y2, F] = model_rpn.predict(X)
                # print(F.shape)
                R = roi_helpers.rpn_to_roi(Y1, Y2, C, overlap_thresh=0.5)
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
                    # print(P_cls.shape[1])
                    for ii in range(P_cls.shape[1]):
                        if np.max(P_cls[0, ii, :]) < bbox_threshold: # or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                            continue
                        cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                        # if cls_name == 'red':
                        #     print(cls_name)
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
                    bbox = np.array(bboxes[key])
                    new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
                    # print(new_boxes.shape[0])
                    for jk in range(new_boxes.shape[0]):
                        (x1, y1, x2, y2) = new_boxes[jk,:]
                        # print(new_boxes[jk,:])
                        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                        all_dets.append((key,100*new_probs[jk]))
                        cen_x = (real_x1 + real_x2)/2
                        cen_y = (real_y1 + real_y2)/2

                        cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (
                        int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)

                        textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                        all_dets.append((key, 100 * new_probs[jk]))

                        (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        textOrg = (real_x1, real_y1 - 0)

                        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

                        tileOP[np.uint16(cen_x), np.uint16(cen_y)] = 255
                        # print(np.uint16(cen_x))
                        # print(np.uint16(cen_y))
            op[row:row + window, col:col + window] = tileOP
            op1[row:row + window, col:col + window, :] = img

    cv2.imwrite(os.path.join(maskOut,files), op)



    print('Elapsed time = {}'.format(time.time() - st))
    # print(all_dets)
    # print(bbox)
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    # cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
