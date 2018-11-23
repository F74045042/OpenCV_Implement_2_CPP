# -*- coding: utf-8 -*-

import sys
from opencv_implement_ui import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication

import numpy as np
import matplotlib.pyplot as plt
import glob


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn2_3.clicked.connect(self.on_btn2_3_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn3_3.clicked.connect(self.on_btn3_3_click)
        self.btn3_4.clicked.connect(self.on_btn3_4_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.comboBox.activated.connect(self.on_combobox_selected)
        self.comboBox.setCurrentIndex(-1)

    #Hide plt toolbar
    plt.rcParams['toolbar'] = 'None'

    # done
    def on_btn1_1_click(self):
        #load image
        img = cv2.imread('images/plant.jpg', 0)

        #show image
        cv2.imshow('Original Image', img)

        #calculate histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        #show image histogram
        plt.bar(range(0, 256), hist, 1) 
        # plt.hist(img.ravel(), 256, [0, 256], facecolor='r')
        plt.xlabel('gray value')
        plt.ylabel('pixel number')
        plt.title('Original image histogram')
        plt.xlim(0,256)
        plt.show()

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    # done
    def on_btn1_2_click(self):
        #load image
        img = cv2.imread('images/plant.jpg', 0)

        #Histogram equalization
        equ = cv2.equalizeHist(img)

        #show image
        cv2.imshow('Equalize image', equ)

        #calculate histogram
        hist = cv2.calcHist([equ], [0], None, [256], [0, 256])

        #show image histogram
        plt.bar(range(0, 256), hist, 1) 
        # plt.hist(equ.ravel(), 256, [0, 256], facecolor='r')
        plt.xlabel('gray value')
        plt.ylabel('pixel number')
        plt.title('Equalize image histogram')
        plt.xlim(0,256)
        plt.show()

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    # done
    def on_btn2_1_click(self):
        #load image and gray image
        img = cv2.imread('images/q2_train.jpg', 0)
        img_out = cv2.imread('images/q2_train.jpg', 1)
        
        #smooth
        kernel_size = (3, 3)
        img = cv2.GaussianBlur(img, kernel_size, 0)

        #hough circles
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,10,param1=50,param2=20,minRadius=10,maxRadius=20)

        #draw circle
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            #for circle
            cv2.circle(img_out,(i[0],i[1]),i[2],(0,255,0),2)
            #for center
            cv2.circle(img_out,(i[0],i[1]),1,(0,0,255),2)
        
        #show image
        cv2.imshow('Input image', img)
        cv2.imshow('Output image',img_out)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    # done
    def on_btn2_2_click(self):
        #load image and gray image
        img = cv2.imread('images/q2_train.jpg', 0)
        img_out = cv2.imread('images/q2_train.jpg', 1)

        #show image
        cv2.imshow('Input image', img_out)
        
        #smooth
        kernel_size = (3, 3)
        img = cv2.GaussianBlur(img, kernel_size, 0)

        #hough circles
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,10,param1=50,param2=20,minRadius=10,maxRadius=20)

        #ROI
        mask = img
        mask[:,:] = 0
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(mask,(i[0],i[1]),i[2],1,-1)

        #convert to hsv
        img_hsv = cv2.cvtColor(img_out,cv2.COLOR_BGR2HSV)

        # calculate histogram
        hist = cv2.calcHist([img_hsv], [0], mask, [180], [0, 180])

        # probability
        hist = hist/hist.max()

        # show image histogram
        plt.bar(range(0, 180), hist) 
        plt.xlabel('Angle')
        plt.ylabel('Probability')
        plt.title('Normalize Hue histogram')
        plt.show()

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    # done
    def on_btn2_3_click(self):
        #load image and gray image
        img = cv2.imread('images/q2_train.jpg', 0)
        img_out = cv2.imread('images/q2_train.jpg', 1)
        
        #smooth
        kernel_size = (3, 3)
        img = cv2.GaussianBlur(img, kernel_size, 0)

        #hough circles
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,10,param1=50,param2=20,minRadius=10,maxRadius=20)

        #ROI
        mask = img
        mask[:,:] = 0
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(mask,(i[0],i[1]),i[2],1,-1)

        #convert to hsv
        img_hsv = cv2.cvtColor(img_out,cv2.COLOR_BGR2HSV)

        #calculate histogram
        hist = cv2.calcHist([img_hsv], [0, 1], mask, [180, 256], [103,121,48,190])

        #normalize
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        #load image and show
        result = cv2.imread('images/q2_test.jpg')
        cv2.imshow('q2_test.jpg', result)

        #convert to hsv
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        #black projection
        dst = cv2.calcBackProject([result_hsv], [0, 1], hist, [103,121,48,190], 1)

        #convolute
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(dst, -1, disc, dst)
        _, dst = cv2.threshold(dst,60,255,0)

        #show image
        cv2.imshow('BackProjection_result.jpg', dst)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    # done
    def on_btn3_1_click(self):
        #load 1~15.bmp and sort
        file = glob.glob('images/CameraCalibration/*.bmp')
        file.sort()
        img = [cv2.imread(i) for i in file]

        #terminal criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for i in range(15):
            #convert to gray
            img_gray = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)

            #find the chess board corner
            ret, corners = cv2.findChessboardCorners(img_gray, (11, 8), None)

            #find accurate point
            corners2 = cv2.cornerSubPix(img_gray, corners, (10, 10), (-1, -1), criteria)

            #draw and display the corners
            img[i] = cv2.drawChessboardCorners(img[i], (11, 8), corners2, ret)
            cv2.imshow('Image', img[i])
            cv2.waitKey(500)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    # done
    def on_btn3_2_click(self):
        #load 1~15.bmp and sort
        file = glob.glob('images/CameraCalibration/*.bmp')
        file.sort()
        img = [cv2.imread(i) for i in file]

        #terminal criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        #Store obj point and image point
        objPoint = []   #3d point in real world
        imgPoint = []   #2d point in image plane

        #prepare object point, (0,0,0) (1,0,0) ... (10,7,0)
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        for i in range(15):
            #convert to gray
            img_gray = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)

            #find the chess board corner
            ret, corners = cv2.findChessboardCorners(img_gray, (11, 8), None)

            if ret == True:
                #add object points
                objPoint.append(objp)

                #find accurate point
                corners2 = cv2.cornerSubPix(img_gray, corners, (10,10), (-1, -1), criteria)

                #add image points
                imgPoint.append(corners2)

                ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoint, imgPoint, img_gray.shape[::-1],None,None)

            print('Image: ',i+1,'.bmp')
            print(intrinsic_mtx)

    # done
    def on_combobox_selected(self, j):
        #global variable for btn3_3
        global extrinsic_mtx

        #load 1~15.bmp and sort
        file = glob.glob('images/CameraCalibration/*.bmp')
        file.sort()
        img = [cv2.imread(i) for i in file]

        #terminal criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        #Store obj point and image point
        objPoint = []   #3d point in real world
        imgPoint = []   #2d point in image plane

        #prepare object point, (0,0,0) (1,0,0) ... (10,7,0)
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        for i in range(15):
            #convert to gray
            img_gray = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)

            #find the chess board corner
            ret, corners = cv2.findChessboardCorners(img_gray, (11, 8), None)

            if ret == True:
                #add object points
                objPoint.append(objp)

                #find accurate point
                corners2 = cv2.cornerSubPix(img_gray, corners, (10,10), (-1, -1), criteria)

                #add image points
                imgPoint.append(corners2)

                ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoint, imgPoint, img_gray.shape[::-1],None,None)

        rmat,_ = cv2.Rodrigues(rvecs[j])
        extrinsic_mtx = np.hstack((rmat,tvecs[j]))

    # done
    def on_btn3_3_click(self):
        print('\n',extrinsic_mtx,'\n')

    # done
    def on_btn3_4_click(self):
        #load 1~15.bmp and sort
        file = glob.glob('images/CameraCalibration/*.bmp')
        file.sort()
        img = [cv2.imread(i) for i in file]

        #terminal criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        #Store obj point and image point
        objPoint = []   #3d point in real world
        imgPoint = []   #2d point in image plane

        #prepare object point, (0,0,0) (1,0,0) ... (10,7,0)
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        for i in range(15):
            #convert to gray
            img_gray = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)

            #find the chess board corner
            ret, corners = cv2.findChessboardCorners(img_gray, (11, 8), None)

            if ret == True:
                #add object points
                objPoint.append(objp)

                #find accurate point
                corners2 = cv2.cornerSubPix(img_gray, corners, (10,10), (-1, -1), criteria)

                #add image points
                imgPoint.append(corners2)

                ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoint, imgPoint, img_gray.shape[::-1],None,None)

            print('Image: ',i+1,'.bmp')
            print(dist)

    # not done yet
    def on_btn4_1_click(self):
        #Global Threshold
        #load image
        img = cv2.imread("images/QR.png", 0)

        ret, img_out = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

        #show image
        cv2.imshow("Original image", img)
        cv2.imshow("Threshold image", img_out)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
