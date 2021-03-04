# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2  
from cv2.ximgproc import guidedFilter
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.io import imread, imsave
import gamma
import mean

imgName = 'cabin-fog'
#I0 = imread(("Images/ori" + imgName +".png"))
#I1 = imread(("Images/fog" + imgName +".png"))

#I0 = imread("Image/" + imgName +".png")
#I1 = imread("Image/" + imgName +".png")

I0 = imread(imgName +".png")
I1 = imread(imgName +".png")

imsave('I0.png', I0)
imsave('I1.png', I1)
# create D1
import dehaze1
I2 = imread("D1.png")

 
# create D2
import dehaze2
I3 = imread("D2.png")


# gamma
im = cv2.imread("D2.png")
G1 = gamma.gamma(im)
I4 = cv2.imread("G1.png")
#I0 = cv2.imread("I0.png")


imsave('I0/'+imgName + 'I0.png', I0)
imsave('I1/'+imgName + 'I1.png', I1)
imsave('I2/'+imgName + 'I2.png', I2)
imsave('I3/'+imgName + 'I3.png', I3)
imsave('I4/'+imgName + 'I4.png', G1)


#PSNR&MAE
print('PSNR I0-I2 :',cv2.PSNR(I0, I2))
print('PSNR I0-I3 :',cv2.PSNR(I0, I3))
print('PSNR I0-I4 :',cv2.PSNR(I0, I4))

#MAE
print('MAE I0-I2 :',mean.color_MAE(I0,I2))
print('MAE I0-I3 :',mean.color_MAE(I0,I3))
print('MAE I0-I4 :',mean.color_MAE(I0,I4))
