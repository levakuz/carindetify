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

def check_circle(imagearr, arrpoints):
    y = imagearr.shape[0]
    x = imagearr.shape[1]
    circle = False
    print(y)
    print(x)
    print(0.2*y)
    print(0.2*x)
    if arrpoints.size != 0:
        for i in range(len(arrpoints[0])):
            print(y - arrpoints[0][i][1])
            print(x - arrpoints[0][i][0])
            if y - arrpoints[0][i][1] < 0.1*y and x - arrpoints[0][i][0] < 0.1*x:
                circle = True


    return circle


def check_size(imagearr, arrpoints):
    x = imagearr.shape[0]
    y = imagearr.shape[1]
    lower = 0.01
    upper = 0.9
    print(y)
    print(x)
    length1 = abs(arrpoints[0][0][0] - arrpoints[0][1][0])
    print(length1)
    length2 = abs(arrpoints[0][3][0] - arrpoints[0][2][0])
    print(length2)
    heigth1 = abs(arrpoints[0][0][1] - arrpoints[0][3][1])
    print(heigth1)
    heigth2 = abs(arrpoints[0][1][1] - arrpoints[0][2][1])
    print(heigth2)
    check = True
    print(x*lower)
    print(y*lower)
    if x*lower > length1 or x*lower > length2 or y*lower > heigth1 or y*lower > heigth2 or x*upper < length1 or x*upper < length2 or y*upper < heigth1 or y*upper < heigth2:
        check = False
    print(check)
    return check


def find_biggest(arrpoint):
    lengths = [0, 0, 0 ]
    print(arrpoint)
    for i in range(len(arrpoint)):
        print(i)
        lengths[i] = (abs(arrpoint[i][0][0] - arrpoint[i][-1][0]) * abs(arrpoint[i][0][1] - arrpoint[i][1][0]))
        print(lengths[i])
    largest = lengths.index(max(lengths))
    return largest


def check_minus(arrPoints):
    sorting = []
    for i in range(len(arrPoints[0]) - 1):
        for j in range(i + 1, len(arrPoints[0])):
            print(int(arrPoints[0][i][0]))
            print(int(arrPoints[0][j][0]))
            if abs(arrPoints[0][i][0] - arrPoints[0][j][0]) < 1:
                print("Есть одинаковые")
                sorting.append(0)

            else:
                sorting.append(1)
                print("Все элементы уникальны")
        # for j in range(i + 1, len(arrPoints[0])):
        #     print(int(arrPoints[0][i][0]))
        #     print(int(arrPoints[0][j][0]))
        #     if abs(arrPoints[0][i][0] - arrPoints[0][j][0]) < 1 and abs(arrPoints[0][i][1] - arrPoints[0][j][1]) < 1:
        #         print("Есть одинаковые")
        #         print(sorting)
        #         sorting.append(False)
        #         break
        #         return sorting
    return sorting


def check_equals(arrPoints):
    sorting = []
    for i in range(len(arrPoints[0]) - 1):
        for j in range(i + 1, len(arrPoints[0])):
            print(int(arrPoints[0][i][0]))
            print(int(arrPoints[0][j][0]))
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
    check = []
    if len(text) >= 5:
        print("1")
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
        print(check)
        return False
    else:

        if len(text) >= 5:
            return text


# def check_number_by(text):
#     check = []
#
#     for i in range(len(text)):
#         if len(text[i]) == 7:
#             for j in range(len(text[i])):
#                 if j in {0,1,2,3,6}:
#                     if text[i][j].isnumeric():
#                         check.append(text[i])
#                 #     else:
#                 #         check.append(False)
#                 # else:
#                 #     if text[i][j] in {'A', 'B', 'E', 'I', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X'}:
#                 #         check.append(True)
#                 #     else:
#                 #         check.append(False)
#
#     if False in check or not check:
#         return False
#     else:
#         for i in range(len(text)):
#             if len(text[i]) == 7:
#                 return text[i]

def add_text(image, text):
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


def read_and_image(image ,folder):
    new_text = 0
    # BLUR = 21
    # CANNY_THRESH_1 = 10
    # CANNY_THRESH_2 = 200
    # MASK_DILATE_ITER = 10
    # MASK_ERODE_ITER = 10
    # MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format
    # im = cv2.imread(image)
    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(imgray, CANNY_THRESH_1, CANNY_THRESH_2)
    # edges = cv2.dilate(edges, None)
    # edges = cv2.erode(edges, None)
    # #cv2.waitKey(100000)
    # contour_info = []
    # contours, hierarchy= cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    # # Previously, for a previous version of cv2, this line was:a
    # #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # # Thanks to notes from commenters, I've updated the code but left this note
    # for c in contours:
    #     contour_info.append((
    #         c,
    #         cv2.isContourConvex(c),
    #         cv2.contourArea(c),
    #     ))
  #  contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
 #   max_contour = contour_info[0]
 #   print(max_contour)
    #cv2.drawContours(im,max_contour,0,(0,255,0), 3)
 #   ellipse = cv2.fitEllipse(max_contour[0])
 #   print(ellipse)
  #  little_ellipse_width = ellipse[1][0]
   # little_ellipse_height = ellipse[1][1]
   # little_ellipse = ((ellipse[0][0],ellipse[0][1]), (little_ellipse_width, little_ellipse_height), ellipse[2])
    #cv2.ellipse(im, little_ellipse, (0, 255, 0), 2)
   # cv2.imshow("1223", im)
   # cv2.waitKey()


    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    #mask = np.zeros(edges.shape)
    #cv2.fillConvexPoly(mask, max_contour[0], (255))
    #cv2.imshow("123",mask)
    #cv2.waitKey()

    # -- Smooth mask, then blur it --------------------------------------------------------
    #mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    #mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    #mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    #mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background --------------------------------------
    #mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    #img = im.astype('float32') / 255.0  # for easy blending

    #masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    #masked = (masked * 255).astype('uint8')  # Convert back to 8-bit

    # cv2.imwrite('img.jpg', im)  # Display
  #  cv2.waitKey()
   # cv2.imshow("123",thresh)
   # cv2.waitKey(20000)
    #cv2.imshow("123",im)
    #cv2.waitKey(20000)

    #print(image[-4:])

    if image[-4:] == '.png':
        new_image = cv2.imread(image)
        cv2.imwrite(image[:-4] + '.jpg',new_image)
        image = image[:-4] + '.jpg'
        print(image)

    if image[-5:] == '.jpeg':
        new_image = cv2.imread(image)
        cv2.imwrite(image[:-5] + '.jpg', new_image)
        image = image[:-5] + '.jpg'
        print(image)
    print(image[10:])
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
                print(arrPoints)
                # 0 = find_biggest(arrPoints)

                # find text with postprocessing by standart
                textArr = textDetector.predict(zones)
                textArr = textPostprocessing(textArr, regionNames)
                print(textArr)
                # textArr = np.sort(textArr, axis=0)
                largest_text = textArr.index(max(textArr, key=len))
                for i in range(len(textArr)):
                    if len(textArr[i]) == 7:
                        largest_text = i
                print(textArr[largest_text])
                new_text = check_number_by(textArr[largest_text])
                if new_text:
                    carimg = add_text(carimg, new_text)
                    # cv2.imshow('123', carimg)
                    # cv2.waitKey()
                    x_tuple = []
                    y_tuple = []
                    # is_circle = check_circle(carimg, arrPoints)
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
                    # splashs = filters.color_splash(img, cv_img_masks)
                    # for splash in splashs:
                    #     plt.imshow(splash)
                    #     plt.axis("off")
                    #     plt.show()

                    # print(is_circle)
                    if arrPoints.size != 0 and arrPoints[0].size < 24:
                        # if arrPoints.size > 8:
                        #     for i in range(arrPoints[0].size):
                        #         summa1= arrPoints[0].sum(axis=1)
                        #         summa2= arrPoints[1].sum(axis=1)
                        #         summa1 = summa1.sum(axis=0)
                        #         summa2 = summa2.sum(axis=0)
                        #     if summa1< summa2:
                        #         arrPoints[0] = arrPoints[1]

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
                                # print(arrPoints[0][i])
                                if arrPoints[0][i][0] < 0:
                                    arrPoints[0][i][0] = 0
                                if arrPoints[0][i][1] < 0:
                                    arrPoints[0][i][1] = 0
                                x_tuple.append(arrPoints[0][i][0])
                                y_tuple.append(arrPoints[0][i][1])
                            # print(x_tuple)
                            # print(y_tuple)
                            up_left = [int(x_tuple[0]), int(y_tuple[0])]
                            # print(up_left)
                            up_right = [int(x_tuple[1]), int(y_tuple[1])]
                            # print(up_right)
                            down_left = [int(x_tuple[3]), int(y_tuple[3])]
                            # print(down_left)
                            down_right = [int(x_tuple[2]), int(y_tuple[2])]
                            # print(down_right)
                            height_carimg, width_carimg, ch1 = carimg.shape
                            #print(x_tuple)

                            logoimg = cv2.imread('./gosnomer_evropa.png')
                            img2gray = cv2.cvtColor(logoimg, cv2.COLOR_BGR2GRAY)
                            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                            mask_inv = cv2.bitwise_not(mask)
                            logoimg = cv2.bitwise_and(logoimg, logoimg, mask=mask)

                            #logoimg = np.dstack([bgr, alpha])  # Add the alpha channe

                            logoimg = cv2.resize(logoimg, (height_carimg, width_carimg))
                            # width = width_carimg
                            # height = height_carimg
                            height_logoimg, width_logoimg, ch = logoimg.shape
                            coef_width = width_logoimg / width_carimg
                            coef_height = height_logoimg / height_carimg

                            # print(height_logoimg)
                            # print(width_logoimg)

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
                            # cv2.imshow('123',carimg)
                            # cv2.waitKey()
                            # splashs = filters.color_splash(carimg, cv_img_masks)
                            # for splash in splashs:
                            #     plt.imshow(splash)
                            #     plt.axis("off")
                            #     plt.show()
                            cv2.imwrite('./' + folder + image[10:], carimg)
                        else:
                            cv2.imwrite('./' + folder + image[10:], carimg)
                    else:
                        cv2.imwrite('./'+ folder + image[10:], carimg)
                else:
                    cv2.imwrite('./'+ folder + image[10:], carimg)
        except:
            print("error")
            cv2.imwrite('./'+ folder + image[10:], carimg)
    else:
        cv2.imwrite('./' + folder + image[10:], carimg)
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
#read_and_image('./by/thumb - 2021-02-03T195017.956.jpeg')
# read_and_image('./by/thumb - 2021-02-03T200211.778.jpeg','test1')
# read_and_image('./by/3909.jpg','test1')
