# coding: utf-8
import cv2
import numpy

#读取图像rgb三通道
img = cv2.imread("./sensitive/s89.jpg")

#读取图像灰度通道
#img_gray = cv2.imread("./sensitive/s89.jpg",cv2.IMREAD_GRAYSCALE)

#分离图像rgb三通道
b,g,r = cv2.split(img)

#打印通道矩阵数值
print(b)
print(r)
print(b-r)

#展示彩色图像和三通道图像
cv2.imshow('img',img)
cv2.imshow("Blue 1",b)
cv2.imshow("Green 1",g)
cv2.imshow("Red 1",r)

#展示灰度图像和通道差值
#cv2.imshow('img_gray',img_gray)
#cv2.imshow("r-g",r-g)

#等待关闭
cv2.waitKey(0)
