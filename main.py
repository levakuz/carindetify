# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import cv2
from scipy.spatial import distance as dist
from operator import itemgetter
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
        x_tuple = []
        y_tuple = []
        """
        #arrPoints = np.sort(arrPoints, axis=1)
        s = arrPoints.sum(axis=2)
        print(s)
        #s = arrPoints.sum(axis=2)
        print(s)
        rect = np.zeros_like(arrPoints)
        rect[0][0] = arrPoints[0][np.argmin(s)]
        rect[0][2] = arrPoints[0][np.argmax(s)]
        diff = np.diff(arrPoints, axis=2)
        print(diff)
        rect[0][1] = arrPoints[0][np.argmin(diff)]
        rect[0][3] = arrPoints[0][np.argmax(diff)]
        print(rect)
        """
        sorting = []
        for i in range(len(arrPoints[0]) - 1):
            for j in range(i + 1, len(arrPoints[0])):
                print(int(arrPoints[0][i][0]))
                print(int(arrPoints[0][j][0]))
                if abs(arrPoints[0][i][0] - arrPoints[0][j][0]) < 1:
                    print("Есть одинаковые")
                    sorting.append(0)
                    break
                else:
                    sorting.append(1)
                    print("Все элементы уникальны")
        sorting.sort()
        if sorting[0] == 1:
            arrPoints[0] = order_points_old(arrPoints[0])
        else:
            if arrPoints[0][0][0] - arrPoints[0][3][0] < 1:
                arrPoints[0] = order_points(arrPoints[0])
            else:
                arrPoints[0] = order_points_old(arrPoints[0])
        print(arrPoints[0])
        for i in range(len(arrPoints[0])):
            #print(arrPoints[0][i])
            if arrPoints[0][i][0] < 0:
                arrPoints[0][i][0] = 0
            if arrPoints[0][i][1] < 0:
                arrPoints[0][i][1] = 0
            x_tuple.append(arrPoints[0][i][0])
            y_tuple.append(arrPoints[0][i][1])

        print(x_tuple)
        print(y_tuple)
        up_left = [int(x_tuple[0]), int(y_tuple[0])]
        print(up_left)
        up_right = [int(x_tuple[1]), int(y_tuple[1])]
        print(up_right)
        down_left = [int(x_tuple[3]), int(y_tuple[3])]
        print(down_left)
        down_right = [int(x_tuple[2]), int(y_tuple[2])]
        print(down_right)
        height_carimg, width_carimg, ch1 = carimg.shape
        #print(x_tuple)

        logoimg = cv2.imread('./gosnomer_ssha.png')
        logoimg = cv2.resize(logoimg, (height_carimg, width_carimg))
        # width = width_carimg
        # height = height_carimg
        height_logoimg, width_logoimg, ch = logoimg.shape
        coef_width = width_logoimg / width_carimg
        coef_height = height_logoimg / height_carimg

        print(height_logoimg)
        print(width_logoimg)

        # print(width)
        # print(height)

        pts1 = np.float32([up_left, up_right, down_left, down_right])
        pts2 = np.float32([[0, 0], [width_logoimg, 0], [0, height_logoimg], [width_logoimg, height_logoimg]])
        M = cv2.getPerspectiveTransform(pts2, pts1)
        dst = cv2.warpPerspective(logoimg, M, (height_logoimg, width_logoimg))
        img2gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        roi = carimg[0:height_carimg, 0:width_carimg]
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(dst, dst, mask=mask)
        dst = cv2.add(img1_bg, img2_fg)

        carimg[0:height_carimg, 0:width_carimg] = dst

        cv2.imwrite('./output' + image[4:], carimg)
        

    return textArr


def order_points(pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]
        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order

        return np.array([tl, tr, br, bl], dtype="float32")


def order_points_old(pts):
        # initialize a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        print("here")
        print(pts)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        print(rect[0])
        rect[2] = pts[np.argmax(s)]
        print(rect[2])
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        print(rect[1])
        rect[3] = pts[np.argmax(diff)]
        print(rect[3])
        # return the ordered coordinates
        return rect

        # ['JJF509', 'RP70012']
#check_king('./img/1200x900n.png')
#read_and_image('./ru/a7d5685.jpg')
read_and_image('./by/CryuOEA4Lf.jpg')
