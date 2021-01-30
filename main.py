# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import cv2

# Import all necessary libraries.
import numpy as np
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../../nomeroff-net')
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  Detector
from NomeroffNet import  filters
from NomeroffNet import  RectDetector
from NomeroffNet import  OptionsDetector
from NomeroffNet import  TextDetector
from NomeroffNet import  textPostprocessing
from NomeroffNet import  ImgGenerator

# load models

def read_and_image(image):

    #print(image[-4:])
    if image[-4:] == '.png':
        new_image = cv2.imread(image)
        cv2.imwrite(image[:-4] + '.jpg',new_image)
        image = image[:-4] + '.jpg'
    rectDetector = RectDetector()

    optionsDetector = OptionsDetector()
    optionsDetector.load("latest")

    textDetector = TextDetector.get_static_module("by")()
    textDetector.load("latest")

    nnet = Detector()
    nnet.loadModel(NOMEROFF_NET_DIR)

    # Detect numberplate
    rootDir = './img/*'
    imgs = [mpimg.imread(img_path) for img_path in glob.glob(rootDir)]

    #img_path = './img/123.jpg'
    img = mpimg.imread(image)
    # Generate image mask.
    cv_imgs_masks = nnet.detect_mask([img])
    #cv2.imshow("123",cv_imgs_masks)

    for cv_img_masks in  cv_imgs_masks:
        # Detect points.

        arrPoints = rectDetector.detect(cv_img_masks)
        print(arrPoints)

    # cut zones
        zones = rectDetector.get_cv_zonesBGR(img, arrPoints, 64, 295)



        # find standart
        regionIds, stateIds, countLines = optionsDetector.predict(zones)
        regionNames = optionsDetector.getRegionLabels(regionIds)
        cv_imgs_masks

        # find text with postprocessing by standart
        textArr = textDetector.predict(zones)
        #textArr = textPostprocessing(textArr, regionNames)
        #print(textArr)

        carimg = cv2.imread(image)
        logoimg = cv2.imread('./gosnomer_ssha.png')
        x_tuple = []
        y_tuple = []
        for i in range(len(arrPoints[0])):
            #print(arrPoints[0][i])
            if arrPoints[0][i][0] < 0:
                arrPoints[0][i][0] = 0
            if arrPoints[0][i][1] < 0:
                arrPoints[0][i][1] = 0
            x_tuple.append(arrPoints[0][i][0])
            y_tuple.append(arrPoints[0][i][1])

        height_carimg, width_carimg, ch1 = carimg.shape

        #print(x_tuple)


        if  int(x_tuple[0]) - int(x_tuple[3]) < 10 or int(x_tuple[1]) - int(x_tuple[2]) < 10 or int(y_tuple[2]) - int(x_tuple[3]) < 10 or int(y_tuple[1]) - int(y_tuple[0]) < 10 :
            print("here")
            logoimg = cv2.resize(logoimg, (height_carimg, width_carimg))
            # width = width_carimg
            # height = height_carimg
            height_logoimg, width_logoimg, ch = logoimg.shape
            coef_width = width_logoimg / width_carimg
            coef_height = height_logoimg / height_carimg
            up_left = [int(x_tuple[3]), int(y_tuple[3])]
            print(up_left)
            up_right =[int(x_tuple[0]), int(y_tuple[0])]
            print(up_right)
            down_left = [int(x_tuple[2]), int(y_tuple[2])]
            print(down_left)
            down_right = [int(x_tuple[1]), int(y_tuple[1])]
            print(down_right)

            print(height_logoimg)
            print(width_logoimg)

        #print(width)
        #print(height)

            pts1 = np.float32([up_left, up_right,down_left, down_right])
            pts2 = np.float32([[0,0],[width_logoimg, 0],[0, height_logoimg], [width_logoimg, height_logoimg]])
            M = cv2.getPerspectiveTransform(pts2, pts1)
            dst = cv2.warpPerspective(logoimg, M,(height_logoimg, width_logoimg))
            img2gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            roi = carimg[0:height_carimg, 0:width_carimg]
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img2_fg = cv2.bitwise_and(dst, dst, mask=mask)
            dst = cv2.add(img1_bg, img2_fg)

            carimg[0:height_carimg,0:width_carimg ] = dst
        else:
            width = int(max(x_tuple)) - int(min(x_tuple))
            height = int(max(y_tuple)) - int(min(y_tuple))
            logoimg = cv2.resize(logoimg,(width, height))
            carimg[int(min(y_tuple)): int(max(y_tuple)), int(min(x_tuple)): int(max(x_tuple))] = logoimg

        cv2.imwrite('./output' + image[4:] , carimg)
        

    return textArr



        # ['JJF509', 'RP70012']
#check_king('./img/1200x900n.png')
#read_and_image('./by/3328.jpeg')