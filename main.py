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
import imutils

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../nomeroff-net')
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  Detector
from NomeroffNet import  filters
from NomeroffNet import  RectDetector
from NomeroffNet import  OptionsDetector
from NomeroffNet import  TextDetector
from NomeroffNet import  textPostprocessing
from NomeroffNet import  ImgGenerator


def check_size(imagearr, arrpoints):
    """Проверка размеров найденного прямоугольника относительно пропорций"""
    x = imagearr.shape[0]
    y = imagearr.shape[1]
    lower = 0.01
    upper = 0.9
    length1 = abs(arrpoints[0][0][0] - arrpoints[0][1][0])
    length2 = abs(arrpoints[0][3][0] - arrpoints[0][2][0])
    heigth1 = abs(arrpoints[0][0][1] - arrpoints[0][3][1])
    heigth2 = abs(arrpoints[0][1][1] - arrpoints[0][2][1])
    check = True
    if x*lower > length1 or x*lower > length2 or y*lower > heigth1 or y*lower > heigth2 or x*upper < length1 or x*upper < length2 or y*upper < heigth1 or y*upper < heigth2:
        check = False
    return check


def find_biggest(arrpoint):
    """Нахождение наибольшего прямоугольника на изображении"""
    lengths = [0, 0, 0]
    for i in range(len(arrpoint)):
        lengths[i] = (abs(arrpoint[i][0][0] - arrpoint[i][-1][0]) * abs(arrpoint[i][0][1] - arrpoint[i][1][0]))
    largest = lengths.index(max(lengths))
    return largest


def check_minus(arrPoints):
    """Проверка координат с разностью меньше 1"""
    sorting = []
    for i in range(len(arrPoints[0]) - 1):
        for j in range(i + 1, len(arrPoints[0])):
            if abs(arrPoints[0][i][0] - arrPoints[0][j][0]) < 1:
                print("Есть одинаковые")
                sorting.append(0)
            else:
                sorting.append(1)
                print("Все элементы уникальны")
    return sorting


def check_equals(arrPoints):
    """Проверка полностью идентичных координат"""
    sorting = []
    for i in range(len(arrPoints[0]) - 1):
        for j in range(i + 1, len(arrPoints[0])):
            if int(arrPoints[0][i][0]) == int(arrPoints[0][j][0]):
                print("Есть одинаковые")
                sorting.append(0)
                break
                return sorting
            else:
                sorting.append(1)
                print("Все элементы уникальны")
    return sorting


def check_number_by(text):
    """Проверка белорусских номеров"""
    check = []
    if len(text) >= 5:
        for j in range(len(text)):
            if text[0].isnumeric():
                if len(text) == 7:
                    if j in {0,1,6}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j == 5:
                        if text[j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 6:
                    if j in {0,1,5}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j == 4:
                        if text[j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 8:
                    if j in {0,1,2,3,7}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    else:
                        if text[j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 5:
                    if j in {0,4}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j == 3:
                        if text[j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
            else:
                if len(text) == 7:
                    if j in {4,5,6}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j in {0,1}:
                        if text[j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 6:
                    if j in {5}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j in {0}:
                        if text[j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 8:
                    if j in {5,6,7}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j in {0,1}:
                        if text[j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 5:
                    if j in {4}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j == 0:
                        if text[j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)

    if False in check or not check:
        return False
    else:

        if len(text) >= 5:
            return text


def check_number_ru(text):
    """Проверка российских номеров"""
    check = []
    if len(text) >= 5:
        for j in range(len(text)):
            if text[0].isnumeric():
                if len(text) == 7:
                    if j in {1, 2, 5, 6}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j == 3:
                        if text[j] in {'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 8:
                    if j in {1, 2, 5, 6, 7}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    else:
                        if text[j] in {'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
            else:
                if len(text) == 7:
                    if j in {5, 6}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j == 4:
                        if text[j] in {'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 8:
                    if j in {1, 2, 3, 6, 7}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    else:
                        if text[j] in {'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 9:
                    if j in {1, 2, 3, 6, 7, 8}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    else:
                        if text[j] in {'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)
                elif len(text) == 5:
                    if j in {0, 4}:
                        if text[j].isnumeric():
                            check.append(True)
                        else:
                            check.append(False)
                    elif j == 3:
                        if text[j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
                            check.append(True)
                        else:
                            check.append(False)

    if False in check or not check:
        return False
    else:

        if len(text) >= 5:
            return text


def add_text(image, text):
    """Добавляет распознанный номер на картинку"""

    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (200, 200)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 3

    # Using cv2.putText() method
    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return (image)


def read_and_image(image, country):
    new_text = 0
    if image[-4:] == '.png':
        new_image = cv2.imread(image)
        cv2.imwrite(image[:-4] + '.jpg',new_image)
        image = image[:-4] + '.jpg'

    if image[-5:] == '.jpeg':
        new_image = cv2.imread(image)
        cv2.imwrite(image[:-5] + '.jpg', new_image)
        image = image[:-5] + '.jpg'

    rectDetector = RectDetector()

    optionsDetector = OptionsDetector()
    optionsDetector.load("latest")
    if country == 'by':
        textDetector = TextDetector.get_static_module("by")()
    elif country == 'ru':
        textDetector = TextDetector.get_static_module("ru")()
    textDetector.load("latest")

    nnet = Detector()
    nnet.loadModel(NOMEROFF_NET_DIR)

    # Detect numberplate
    rootDir = './img/*'
    imgs = [mpimg.imread(img_path) for img_path in glob.glob(rootDir)]

    #img_path = './img/123.jpg'
    img = mpimg.imread(image)
    # Generatce image mask.
    cv_imgs_masks = nnet.detect_mask([img])
    #cv2.imshow("123",cv_imgs_masks)
    carimg = cv2.imread(image)
    if cv_imgs_masks:
        try:
            for cv_img_masks in  cv_imgs_masks:
                # Detect points.

                arrPoints = rectDetector.detect(cv_img_masks)



            # cut zones
                zones = rectDetector.get_cv_zonesBGR(img, arrPoints, 64, 295)




                # find standart
                regionIds, stateIds, countLines = optionsDetector.predict(zones)
                regionNames = optionsDetector.getRegionLabels(regionIds)
                # 0 = find_biggest(arrPoints)

                # find text with postprocessing by standart
                textArr = textDetector.predict(zones)
                textArr = textPostprocessing(textArr, regionNames)
                largest_text = textArr.index(max(textArr, key=len))
                for i in range(len(textArr)):
                    if len(textArr[i]) == 7:
                        largest_text = i
                if country == 'by':
                    new_text = check_number_by(textArr[largest_text])
                elif country == 'ru':
                    new_text = check_number_ru(textArr[largest_text])
                if new_text:
                    x_tuple = []
                    y_tuple = []
                    if arrPoints.size != 0 and arrPoints[0].size < 24:
                        sorting = check_minus(arrPoints)
                        same = sorting.count(0)
                        if same == 0 or same == 2:
                            arrPoints[0] = order_points_old(arrPoints[0])
                        else:
                            if arrPoints[0][0][0] - arrPoints[0][3][0] < 1:
                                arrPoints[0] = order_points(arrPoints[0])
                            else:
                                arrPoints[0] = order_points_old(arrPoints[0])

                        size_check = check_size(carimg, arrPoints)

                        if size_check:
                            for i in range(len(arrPoints[0])):
                                if arrPoints[0][i][0] < 0:
                                    arrPoints[0][i][0] = 0
                                if arrPoints[0][i][1] < 0:
                                    arrPoints[0][i][1] = 0
                                x_tuple.append(arrPoints[0][i][0])
                                y_tuple.append(arrPoints[0][i][1])
                            up_left = [int(x_tuple[0]), int(y_tuple[0])]
                            up_right = [int(x_tuple[1]), int(y_tuple[1])]
                            down_left = [int(x_tuple[3]), int(y_tuple[3])]
                            down_right = [int(x_tuple[2]), int(y_tuple[2])]
                            height_carimg, width_carimg, ch1 = carimg.shape

                            logoimg = cv2.imread('./gosnomer_evropa.png')
                            img2gray = cv2.cvtColor(logoimg, cv2.COLOR_BGR2GRAY)
                            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                            mask_inv = cv2.bitwise_not(mask)
                            logoimg = cv2.bitwise_and(logoimg, logoimg, mask=mask)

                            #logoimg = np.dstack([bgr, alpha])  # Add the alpha channe

                            logoimg = cv2.resize(logoimg, (height_carimg, width_carimg))

                            height_logoimg, width_logoimg, ch = logoimg.shape
                            coef_width = width_logoimg / width_carimg
                            coef_height = height_logoimg / height_carimg

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
                            cv2.imwrite('./' + image[0:-4] + '.jpeg', carimg)
                        else:
                            cv2.imwrite('./' + image[0:-4] + '.jpeg', carimg)
                    else:
                        cv2.imwrite('./' + image[0:-4] + '.jpeg', carimg)
                else:
                    cv2.imwrite('./' + image[0:-4] + '.jpeg', carimg)
        except:
            print("error")
            cv2.imwrite('./' + image[0:-4] + '.jpeg', carimg)
    else:
        cv2.imwrite('./' + image[0:-4] + '.jpeg', carimg)
    return new_text


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
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect
