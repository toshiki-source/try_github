# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
from skimage.io import imread, imsave

im_name = "maguey"


# Create D1
#os.system("python dehazing-locally_adaptive_test.py -i " + im_name + " -o " + im_name + "_D1")

# Create M1 and M2
#os.system("python canny.py -i " + im_name)

# Create the background for the masked image I2

background = Image.new('1', Image.open(im_name + '.png').size)  # Black background
#cv2.imwrite('background.png', background)
background.save("background.png", "PNG")

# Combine the background with the D1 image with the mask M2
I0 = cv2.imread(im_name + '.png') # Original image
I1 = cv2.imread(im_name + '_D1.png') # Image D1
I7 = cv2.imread('M2.png') # Mask M2

#I2 = Image.composite(I1, background, I7)
I2 = cv2.bitwise_and(I1, I7)
cv2.imwrite('I2.png', I2)

os.system("python dehazing-locally_adaptive_test.py -i I2 -o I3")
print('A')
I3 = Image.open('I3.png') # Dehaze x2 + Mask M2
print('B')
I6 = Image.open('M1.png').convert('L') # Mask M1
print('C')
I6_blur = I6.filter(ImageFilter.GaussianBlur(10)) # Blur mask M1
print('D')
cv2.imshow('sampleI1', I1)
cv2.imshow('sampleI3', I3)
cv2.imshow('sampleI6', I6_blur)
I8 = Image.composite(I1, I3, I6_blur)  # Output image
print('E')
cv2.imwrite('I8.png', I8)
