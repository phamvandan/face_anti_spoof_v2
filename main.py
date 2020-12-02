"""
This script provides a local test routine so you can verify the algorithm works before pushing it to evaluation.

It runs your detector on several local images and verify whether they have obvious issues, e.g:
    - Fail to start
    - Wrong output format

It also prints out the runtime for the algorithms for your references.


The participants are expected to implement a face forgery detector class. The sample detector illustrates the interface.
Do not modify other part of the evaluation toolkit otherwise the evaluation will fail.

Author: Yuanjun Xiong, Zhengkui Guo, Yuanhan Zhang
Contact: celebaspoof@gmail.com

CelebA-Spoof 
"""
import cv2
import os
import time
import sys
import logging
import pandas as pd
import numpy as np
from client import get_image, verify_output
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from datetime import datetime
from  client import  LOCAL_IMAGE_LIST_PATH, read_image
logging.basicConfig(level=logging.INFO)
model_dir="./resources/anti_spoof_models"
model_test = AntiSpoofPredict(0)
image_cropper = CropImage()

from tsn_predict import TSNPredictor as CelebASpoofDetector


def run_test(detector_class):
    """
    In this function we create the detector instance. And evaluate the wall time for performing CelebASpoofDetector.
    """

    # initialize the detector
    logging.info("Initializing detector.")
    try:
        detector = detector_class()
    except:
        # send errors to the eval frontend
        raise
    logging.info("Detector initialized.")


    # run the images one-by-one and get runtime
    output_probs = {}
    eval_cnt = 0
    real=0
    not_de=0
    tong=0
    results=[]
    logging.info("Starting runtime evaluation")
    image_list = os.listdir(LOCAL_IMAGE_LIST_PATH)
    # image_list=[]
    # for img_path in list_img:
    #     image_list.append(os.path.join(img_folder,img_path))

    for idx, image_id in enumerate(image_list):
        # get image from local file
        try:
            image, bbox, ori_img = read_image(image_id)
            prob = detector.predict(image)
            print(prob)
            # for idx, i in enumerate(image_id):
            # output_probs[i] = float(prob[0][1])
            tong=tong+1
            if prob[0][1] >= 0.81 and prob[0][0]<0.4  :
                print(image_id," is fake "," : ", float(prob[0][1]))
                results.append([image_id, "f1", float(prob[0][1])])
            else:
                # print(ori_image.shape)
                prediction = np.zeros((1, 3))
                test_speed = 0
                # sum the prediction from single model's result
                for model_name in os.listdir(model_dir):
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    param = {
                        "org_img": ori_img,
                        "bbox": bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    if scale is None:
                        param["crop"] = False
                    img = image_cropper.crop(**param)
                    start = time.time()
                    prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                    test_speed += time.time() - start
                print("Prediction cost {:.2f} s".format(test_speed))
                label = np.argmax(prediction)
                value = prediction[0][label] / 2
                if not label == 1:
                    print("fake", image_id, "confidence=", value)
                    results.append([image_id, "f2", float(value)])
                else:
                    if value <= 0.67:
                        print("fake", image_id, "confidence=", value)
                        results.append([image_id, "f2", float(value)])
                    else:
                        print("real", image_id, "confidence=", value)
                        real = real+1
                        results.append([image_id, "r", float(value)])
        except:
            # send errors to the eval frontend
            logging.error("Image id failed: {}".format(image_id))
            raise
        eval_cnt += len(image)

        if eval_cnt % 10 == 0:
            logging.info("Finished {} images".format(eval_cnt))
    save_dir="save_dir"
    try :
        os.mkdir(save_dir)
    except OSError:
        pass
    pd.DataFrame(results, columns=["path_to_image", "fake_real", "conf"]).to_csv(
        os.path.join(save_dir, "{}_result.csv".format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))))

    print("so anh real: ",real," tren tong so : ",tong)
    print("so anh not_detect: ",not_de," tren tong so : ",tong)
    logging.info("""
    ================================================================================
    All images finished, showing verification info below:
    ================================================================================
    """)

    # verify the algorithm output
    # verify_output(output_probs)


if __name__ == '__main__':
    run_test(CelebASpoofDetector)
