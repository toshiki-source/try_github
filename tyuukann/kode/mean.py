# -*- coding: utf-8 -*-

# CSVを読み込み用
import pandas as pd
# Mean Absolute Error(MAE)用
from sklearn.metrics import mean_absolute_error
# Root Mean Squared Error(RMSE)用
from sklearn.metrics import mean_squared_error
import numpy as np


def color_MAE(im1,im2):


   y0 = mean_absolute_error(im1[:,:,0], im2[:,:,0])
   y1= mean_absolute_error(im1[:,:,1], im2[:,:,1])
   y2= mean_absolute_error(im1[:,:,2], im2[:,:,2])
   mae = (y0 + y1 + y2) / 3
   
  # print('MAE : {:.3f}'.format(mae))
   return mae
