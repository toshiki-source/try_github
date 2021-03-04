# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2  
from skimage.measure import compare_psnr, compare_ssim
from cv2.ximgproc import guidedFilter
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.io import imread, imsave
import gamma
import mean

count = '30'
filName = '703'
imgName = '703_9_0.74287'

#I0 = imread(("Images/ori" + imgName +".png"))
#I1 = imread(("Images/fog" + imgName +".png"))

#I0 = imread("Image/" + imgName +".png")
#I1 = imread("Image/" + imgName +".png")

I0 = imread('clear/'+ filName +".png")
I1 = imread('01hazy/'+ imgName +".png")
imsave('I0/'+ filName + 'I0.png', I0)
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


#imgsave
imsave('I0/'+ count + 'I0.png', I0)
imsave('I1/'+ count + 'I1.png', I1)
imsave('I2/'+ count + 'I2.png', I2)
imsave('I3/'+ count + 'I3.png', I3)
imsave('I4/'+ count + 'I4.png', G1)





#MAE
#print('MAE I0-I2 :',mean.color_MAE(I0,I2))
#print('MAE I0-I3 :',mean.color_MAE(I0,I3))
#print('MAE I0-I4 :',mean.color_MAE(I0,I4))

#SSIM
SSIMI0I1 = compare_ssim(I0,I1, multichannel=True)
SSIMI0I2 = compare_ssim(I0,I2, multichannel=True)
SSIMI0I3 = compare_ssim(I0,I3, multichannel=True)
SSIMI0I4 = compare_ssim(I0,G1, multichannel=True)

##print(SSIMI0I1)
##print(SSIMI0I2)
##print(SSIMI0I3)
##print(SSIMI0I4)
print(SSIMI0I1, SSIMI0I2, SSIMI0I3, SSIMI0I4)

#PSNR
#print('PSNR I0-I1 :',cv2.PSNR(I0, I1))
#print('PSNR I0-I2 :',cv2.PSNR(I0, I2))
#print('PSNR I0-I3 :',cv2.PSNR(I0, I3))
#print('PSNR I0-I4 :',cv2.PSNR(I0, G1))
print(cv2.PSNR(I0, I1), cv2.PSNR(I0, I2), cv2.PSNR(I0, I3), cv2.PSNR(I0, G1))

print(count)
#print('SSIM I0-I2 :',compare_ssim(I0,I2, multichannel=True))
#print('SSIM I0-I3 :',compare_ssim(I0,I3, multichannel=True))
#print('SSIM I0-I4 :',compare_ssim(I0,I4, multichannel=True))
#print('SSIM I0-I4 :',compare_ssim(I0,G1, multichannel=True))








