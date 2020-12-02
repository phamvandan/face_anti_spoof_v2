import json
import cv2
import os
import time
import sys
import imutils

import logging

import numpy as np
from retina.detect import load_net, do_detect

file =open("file_path.txt","r")
path_folder=file.readlines()
file.close()
#================================================================================
# Please change following path to your OWN
LOCAL_ROOT = path_folder[0].replace('\n','')+"/"
LOCAL_IMAGE_LIST_PATH = path_folder[0].replace('\n','')
#================================================================================

net,device,cfg=load_net()
img_heights =[800, 1500]
def check_box_angle(landmarks):
    y1 = landmarks[1]
    y2 = landmarks[3]
    y = landmarks[5]
    if y1 < y and y2 < y:
        return 0
    elif y1 < y < y2:
        return 270
    elif y1 > y and y2 > y:
        return 180
    elif y2 < y < y1:
        return 90
    print("UNKNOWN")
    return 0

def rotate_box(bbox, angle, h, w):
    x1, y1, x2, y2, conf = bbox
    if angle == 0:
        return bbox
    elif angle == 90:
        return w - y2, x1, w - y1, x2, conf
    elif angle == 180:
        return w - x2, h - y2, w - x1, h - y1, conf
    else:
        return y1, h - x2, y2, h - x1, conf

def rotate_image(image, angle):
    image_rs = None
    if angle == 0:
        image_rs = image
    if angle == 90:
        image_rs = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        image_rs = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        image_rs = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_rs

def faceboxes_detect(image, img_heights, exact_thresh):
    box = None
    old_conf = 0.5
    image_rs = None
    angle = None
    resize_w = None
    resize_h = None
    landmark = None
    for img_height in img_heights:
        img = imutils.resize(image, height=img_height)
        for i in range(4):
            if i != 0:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            bboxs, landmarks = do_detect(img, net, device, cfg)
            if bboxs is None or len(bboxs) == 0:
                continue
            idx = np.argmax(bboxs[:, 4])
            bbox = bboxs[idx]
            if bbox[-1] > old_conf:
                old_conf = bbox[-1]
                box = bbox
                angle = 90 * i
                resize_h, resize_w = img.shape[:2]
                landmark = landmarks[idx]

            if old_conf > exact_thresh:
                break

    if box is not None:
        image_rs = rotate_image(image, angle)
        ori_h, ori_w = image_rs.shape[:2]
        x, y, a, b, conf = box
        box = [int(x * ori_w / resize_w), int(y * ori_h / resize_h), int(a * ori_w / resize_w),
               int(b * ori_h / resize_h), conf]
        # x, y, a, b, conf = box
        # cv2.rectangle(image_rs, (x, y), (a, b), (0, 0, 255), 2)
        # cv2.imshow("image_rs", image_rs)
        # cv2.waitKey(0)
        angle = check_box_angle(landmark)
        print("angle", angle)
        image_rs = rotate_image(image_rs, angle)
        ori_h, ori_w = image_rs.shape[:2]
        box = list(rotate_box(box, angle, ori_h, ori_w))
        x, y, a, b, _ = box

        box[2] = a - x
        box[3] = b - y

        # cv2.rectangle(image_rs, (x, y), (a, b), (0, 0, 255), 2)
        # cv2.imshow("image_rs", image_rs)
        # cv2.waitKey(0)
    return image_rs, box


def read_image(image_path):
    """
    Read an image from input path

    params:
        - image_local_path (str): the path of image.
    return:
        - image: Required image.
    """

    image_path = LOCAL_ROOT + image_path

    img = cv2.imread(image_path)
    # Get the shape of input image
    real_h,real_w,c = img.shape
    #assert os.path.exists(image_path[:-4] + '_BB.txt'),'path not exists' + ' ' + image_path
    #rects, landms = do_detect(img, net, device, cfg)
    img, rects = faceboxes_detect(img, img_heights, exact_thresh=0.8)
    ori_img = img.copy()
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    print(rects)
    #print(rects)
    try:
        x,y,a,b,score = rects
        w = x+a
        h = y+b
    except:
        logging.info('Bounding Box of' + ' ' + image_path + ' ' + 'is wrong')   

    try:
        w = int(float(w))+int((float(w)-float(x))/6.0)
        #print(w)
        h = int(float(h))+int((float(h)-float(y))/6.0)
        x = int(float(x))-int((float(w)-float(x))/6.0)
        y = int(float(y))-int((float(h)-float(y))/6.0)
        # # # Crop face based on its bounding box
        y1 = 0 if y < 0 else y
        x1 = 0 if x < 0 else x 
        y2 = real_h if h > real_h else h
        x2 = real_w if w > real_w else w
        img = img[y1:y2,x1:x2,:]
        # if y<0:
        #     y1=0
        # else y1=y
        # if x<0:
        #     x1=0
        # else x1=x
        # if h>real_h:
        #     y2=real_h
        # else y2=h
        # if w>real_w:
        #     x2=real_w

        #img = img[y:h,x:w,:]

    except:
        logging.info('Cropping Bounding Box of' + ' ' + image_path + ' ' + 'goes wrong')   
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, rects, ori_img



def get_image(img_folder=LOCAL_IMAGE_LIST_PATH ):
    """
    This function returns a iterator of image.
    It is used for local test of participating algorithms.
    Each iteration provides a tuple of (image_id, image), each image will be in RGB color format with array shape of (height, width, 3)
    
    return: tuple(image_id: str, image: numpy.array)
    """
    # with open(LOCAL_ROOT+LOCAL_IMAGE_LIST_PATH) as f:
    #     image_list = json.load(f)
    # logging.info("got local image list, {} image".format(len(image_list.keys())))
    image_list=os.listdir(img_folder)
    # image_list=[]
    # for img_path in list_img:
    #     image_list.append(os.path.join(img_folder,img_path))
    Batch_size =50
    logging.info("Batch_size= {}".format(Batch_size))
    n = 0
    final_ori_image = []
    final_image_bbox = []
    final_image = []
    final_image_id = []
    for idx,image_id in enumerate(image_list):
        # get image from local file
        try:
            image, bbox, ori_img = read_image(image_id)
            final_image_bbox.append(bbox)
            final_ori_image.append(ori_img)
            final_image.append(image)
            final_image_id.append(image_id)
            n += 1
        except:
            logging.info("Failed to read image: {}".format(image_id))
            raise

        if n == Batch_size or idx == len(image_list) - 1:
            np_final_image_id = np.array(final_image_id)
            np_final_image = np.array(final_image)
            np_final_image_bbox = np.array(final_image_bbox)
            np_final_ori_image = np.array(final_ori_image)
            n = 0
            final_image = []
            final_image_id = []
            final_image_bbox = []
            final_ori_image = []
            yield np_final_image_id, np_final_image, np_final_image_bbox, np_final_ori_image



#Get the threshold under several fpr
def get_thresholdtable_from_fpr(scores,labels, fpr_list):
    """Calculate the threshold score list from the FPR list

    Args:
      score_target: list of (score,label)

    Returns:
      threshold_list: list, the element is threshold score calculated by the
      corresponding fpr
    """
    threshold_list = []
    live_scores = []
    for score, label in zip(scores,labels):
        if label == 0:
            live_scores.append(float(score))
    live_scores.sort(reverse=True)
    live_nums = len(live_scores)
    for fpr in fpr_list:
        i_sample = int(fpr * live_nums)
        i_sample = max(1, i_sample)
        threshold_list.append(live_scores[i_sample - 1])
    return threshold_list

#Get the threshold under thresholds
def get_tpr_from_threshold(scores,labels, threshold_list):
    """Calculate the recall score list from the threshold score list.

    Args:
      score_target: list of (score,label)
      threshold_list: list, the threshold list

    Returns:
      recall_list: list, the element is recall score calculated by the
                   correspond threshold
    """
    tpr_list = []
    hack_scores = []
    for score, label in zip(scores,labels):
        if label == 1:
            hack_scores.append(float(score))
    hack_scores.sort(reverse=True)
    hack_nums = len(hack_scores)
    for threshold in threshold_list:
        hack_index = 0
        while hack_index < hack_nums:
            if hack_scores[hack_index] <= threshold:
                break
            else:
                hack_index += 1
        if hack_nums != 0:
            tpr = hack_index * 1.0 / hack_nums
        else:
            tpr = 0
        tpr_list.append(tpr)
    return tpr_list



def verify_output(output_probs):
    """
    This function prints the groundtruth and prediction for the participant to verify, calculates average FPS.

    params:
    - output_probs (dict): dict of probability of every video
    - output_times (dict): dict of processing time of every video
    - num_frames (dict): dict of number of frames extracting from every video
    """
    with open (LOCAL_ROOT+LOCAL_IMAGE_LIST_PATH,'r') as f:
        gts = json.load(f)

    scores = []
    labels = []
    for k in output_probs:
        #import pdb;pdb.set_trace()
        if k in gts:
            scores.append(output_probs[k])
            # 43 is the index of Live/Spoof label
            labels.append(gts[k][43])

    fpr_list = [0.01, 0.005, 0.001]
    threshold_list = get_thresholdtable_from_fpr(scores,labels, fpr_list)
    tpr_list = get_tpr_from_threshold(scores,labels, threshold_list)
      
    # Show the result into score_path/score.txt  
    logging.info('TPR@FPR=10E-3: {}\n'.format(tpr_list[0]))
    logging.info('TPR@FPR=5E-3: {}\n'.format(tpr_list[1]))
    logging.info('TPR@FPR=10E-4: {}\n'.format(tpr_list[2]))

    logging.info("Done")


