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
from datetime import datetime
logging.basicConfig(level=logging.INFO)


from tsn_predict import TSNPredictor as CelebASpoofDetector


def run_test(detector_class, image_iter):
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
    for image_id, image in image_iter:
        time_before = time.time()
        try:
            prob = detector.predict(image)
            for idx,i in enumerate(image_id):
                output_probs[i] = float(prob[idx][1])
                tong=tong+1
                if prob[idx][1] >= 0.81 and prob[idx][0]<0.4  :
                    print(image_id[idx]," is fake "," : ",float(prob[idx][1]))
                    results.append([image_id[idx], 0, float(prob[idx][1])])
                    
                elif prob[idx][0]> 0.4 and prob[idx][1]<0.4  :
                    print(image_id[idx]," is real "," : ",float(prob[idx][0]))
                    real=real+1
                    results.append([image_id[idx], 1, float(prob[idx][0])])
                    # cv2.imshow("image",image[idx])
                    # cv2.waitKey(0)
                else :
                    print(image_id[idx]," not detect "," : ",float(prob[idx][1]))
                    results.append([image_id[idx], 2, float(prob[idx][0])])
                    not_de=not_de+1
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
    celebA_spoof_image_iter = get_image()
    run_test(CelebASpoofDetector, celebA_spoof_image_iter)
