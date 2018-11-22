# -*- coding: utf-8 -*-

import sys
from opencv_implement_ui import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication

import numpy as np
import matplotlib.pyplot as plt


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

    #Hide plt toolbar
    plt.rcParams['toolbar'] = 'None'

    def on_btn1_1_click(self):
        #load image
        img = cv2.imread('images/plant.jpg', 0)

        #show image
        cv2.imshow('Original Image', img)

        #calculate histogram
        # hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        #show image histogram
        # plt.bar(range(0, 257), hist) 
        # plt.hist(img.ravel(), 256, [0, 256], facecolor='r')
        # plt.xlabel('gray value')
        # plt.ylabel('pixel number')
        # plt.title('Original image histogram')
        # plt.xlim(0,256)
        plt.show()

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn1_2_click(self):
        #load image
        img = cv2.imread('images/plant.jpg', 0)

        #Histogram equalization
        equ = cv2.equalizeHist(img)

        #show image
        cv2.imshow('Equalize image', equ)

        #show image histogram
        plt.hist(equ.ravel(), 256, [0, 256], facecolor='r')
        plt.xlabel('gray value')
        plt.ylabel('pixel number')
        plt.title('Equalize image histogram')
        plt.xlim(0,256)
        plt.show()

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

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

        #calculate histogram
        #hist = cv2.calcHist([img_hsv], [0], mask, [256], [0, 256])

        #probability
        #hist = hist/hist.max()

        #show image histogram
        #plt.bar(range(0, 257), hist) 
        #plt.show()

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()


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

        # #calculate histogram
        # hist = cv2.calcHist([img_hsv], [0], mask, [180, 256], [0,180,0,256])

        # #normalize
        # cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        # #load image and show
        # result = cv2.imread('images/q2_test.jpg')
        # cv2.imshow('q2_test.jpg', result)

        # #convert to hsv
        # result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        # #black projection
        # dst = cv2.calBlackProject([result_hsv], [0, 1], hist, [0, 180, 0, 256], 1)

        # #convolute
        # disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # cv2.filter2D(dst, -1, disc, dst)
        # _, dst = cv2.threshold(res,60,255,0)

        # #show image
        # cv2.imshow('BackProjection_result.jpg', dst)

    def on_btn3_1_click(self):
        print('3.1 clicked')

    def on_btn3_2_click(self):
        print('3.2 clicked')

    def on_btn3_3_click(self):
        print('3.3 clicked')

    def on_btn3_4_click(self):
        print('3.4 clicked')

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
