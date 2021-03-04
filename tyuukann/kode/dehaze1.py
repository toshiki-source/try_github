# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

##############################################################################
## V.H. Diaz-Ramirez, J.E. Hernandez-Beltran, R. Juarez-Salazar, 
## "Real-time haze removal in monocular images using locally adaptive processing,"
##  Journal of Real-Time Image Processing, (2017)
##  DOI:10.1007/s11554-017-0698-z
##############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2  
from cv2.ximgproc import guidedFilter
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.io import imread, imsave

##############################################################################
# nxn領域から中心に近いk個のピクセル抽出
##############################################################################
def KNB(mw,K):
    #print("mw[:,:,0] : ", mw[:,:,0]) # 9x9x3 matrix
    #print("K : ", K) # ? 61
    mwg = rgb2gray(mw)  # グレイスケール化(パラメーターが二次元？) 
    #print("mwg : ", mwg) # - 1 dimension 9x9x1 grayscale
    r = np.size(mwg)    # 配列の中の要素数
    #print("r : ", r) # 81 (9*9)
    
    re,co = np.shape(mwg)   #
    #print("re : ", re)
    #print("co : ", co)
    nbh = np.zeros([re,co,3])  # make [re*co*3] array (row,colum,RBG)
    #print("nbh : ", nbh) # like mw but empty
    #nbh_t = zeros([r,3])
    nbh_v = np.zeros([K,3])    # K*3の配列
    cent = mwg[int(np.floor(re/2)), int(np.floor(co/2))]  # np.floor = 切り捨て
    dist = np.zeros(r)    # 配列の中の要素数を零に
    #print("dist : ", dist) # 
    dist = abs(mwg - cent) # abs=数の絶対値
    ("dist : ", dist) # Color change compared to the center color
    dist_ord = np.sort(np.ravel(dist[:])) # 多次元配列distの中の全てを一次元配列に直してソート
    #print("dist_ord : ", dist_ord) # List from the smallest to the biggest
    dist_K = dist_ord[K-1]
    #print("dist_K : ", dist_K) # in the list, the value number K (here 61)
    x,y = np.where(dist <= dist_K) # dist <= dist_Kの条件でindex取得（そこの画素値）
    #print("x : ", x)
    #print("y : ", y)
    nbh[x,y,0] = mw[x,y,0] # 多次元配列x*y*0  の値を nbh に代入
    #print("nbh[x,y,0] : ", nbh[x,y,0])
    nbh[x,y,1] = mw[x,y,1] # 多次元配列x*y*1
    nbh[x,y,2] = mw[x,y,2] # 多次元配列 x*y*3
    nbh_t0 = sorted(np.ravel(nbh[:,:,0]),reverse=True) # 配列(:*:*0)の画素値を降順で並び替え
    nbh_t1 = sorted(np.ravel(nbh[:,:,1]),reverse=True) # 配列(:*:*1)の画素値を降順で並び替え
    nbh_t2 = sorted(np.ravel(nbh[:,:,2]),reverse=True) # 配列(:*:*2)の画素値を降順で並び替え
    nbh_v[:,0] = nbh_t0[0:K] # 配列nbh_t0を0からKまでの範囲で選択（MINのやつ)
    nbh_v[:,1] = nbh_t1[0:K] # 配列nbh_t1を0からKまでの範囲で選択（MINのやつ)
    nbh_v[:,2] = nbh_t2[0:K] # 配列nbh_t2を0からKまでの範囲で選択（MINのやつ)
    return nbh_v
    

def rgb2gray(img):
    img_gray = np.double( (0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2])) # RGB to Gray Conversion
    return img_gray
        
S = 19 # Defines the size of sliding-window (SxS) for Airlight estimation


### Construction of a synthetic image degraded with haze 合成画像構築
#im = 255*double( imread("undegraded.png") ) # 霧なし画像
#gt = (255.0 - double( 255*imread('depth_ground_truth.png') )) #深度関数d(x)
#r,c,p = im.shape
#f_c = zeros([r,c,3])
#beta =  0.9 / 255.0# Dispef_crsion coeficient 
#t0 = exp(-beta*gt) # 媒体透過関数 式(2)
#C = round(0.95*255) # Ambient light　四捨五入
#f_c[:,:,0] = im[:,:,0]*t0 + C*(1-t0) # Hazy image # 合成霧画像の計算　式(1)
#f_c[:,:,1] = im[:,:,1]*t0 + C*(1-t0) # Hazy image
#f_c[:,:,2] = im[:,:,2]*t0 + C*(1-t0) # Hazy image
#f_sin = im[:,:,0:3]                  # ここ意味わかりません

## Read the input hazy image from drive
# imgName = 'maguey_output00'  # 画像の指定
f_c = np.double( imread("I1.png") ) # Read Input Hazy Image

####################
# Size of the Input Image
Nr,Nc,Np = f_c.shape # 3次元配列の要素数（合成霧画像）

# Nr = 640,Nc = 512.
#########################################
# Extention of the input image to avoid edge issues #エッジ問題回避のための入力画像拡張
A1 = np.concatenate((np.flipud(np.fliplr(f_c)), np.flipud(f_c), np.flipud(np.fliplr(f_c))), axis=1) # （転置、上下、転置）
A2 = np.concatenate((np.fliplr(f_c),            f_c,            np.fliplr(f_c)), axis=1) # （左右、そのまま、左右）
A3 = np.concatenate((np.flipud(np.fliplr(f_c)), np.flipud(f_c), np.flipud(np.fliplr(f_c))), axis=1) # （転置、上下、転置）
### fliplr = 配列左右反転
### flipud = 配列上下反転
### flipud(fliplr) = 配列の転置
### concatnateで配列の連結ができる　axis=0で行方向に連結（縦方向） axis=1で列方向に連結（横方向）
f_proc = np.concatenate( (A1,A2,A3) ,axis=0)
f_proc = f_proc[Nr-int((S-1)/2):2*Nr+int((S-1)/2), Nc-int((S-1)/2):2*Nc+int((S-1)/2),:]

A_test = np.zeros([Nr,Nc])
f_mv = np.zeros([S,S])
K = np.floor(2*S*S/3) # floor = 切り捨て
for row in range(Nr):
    leyend = 'Estimating Airlight: ' + str(int(100*(row+1)/Nr)) + '%' # Estimating Airlight:37%
    print(leyend)
    for column in range(Nc):
        f_mv = ( f_proc[ row:S+row, column:S+column ] ) # proc[240~259,512~531]
        #print("f_mv", f_mv)
        #print("f_mv shape", np.shape(f_mv))
        f_max = f_mv.max() # maximum of color in the moving window
        f_min = f_mv.min() # minimum of color in the moving window
        #u = np.mean(f_mv[:]) # 配列平均値
        u = (f_min + f_max) / 2.0 # Average color
        #v = (1 + np.var(f_mv[:]))
        v = f_max - f_min # Difference of color in the moving window
        #print("f_max : ", f_max, ", f_min : ", f_min, ", u : ", u, ", v : ", v)
        A_test[row,column] = u / (1 + v)
	#print(A_test)
	#quit()
print('Done!')

x0,y0 = np.where( A_test == A_test.max()) # (A_test == A_test.max())の部分の画素値取得
A = np.zeros(3)
A[0] = f_c[x0[0], y0[0],0]
A[1] = f_c[x0[0], y0[0],1]
A[2] = f_c[x0[0], y0[0],2]
A_est = 0.2989*A[0] + 0.5870*A[1] + 0.1140*A[2]

t_est = np.zeros([Nr,Nc])
trans = np.zeros([Nr,Nc])

#PAR = [15, 0.01, 1.0, 4.] # Parameters for maguey アロエ
PAR = [15, 0.01, 10.0, 4.0] # Parameters for flores 花
#PAR = [19, 0.6, 0.6, 6.] # Parameters for fuente 最初

S = PAR[0]      # Defines the size of sliding-window SxS スライディングウインドウの設定S×Sの設定
w = PAR[1]      # Minimum allowed value of transmission
j0 = PAR[2]     # Parameter for transmission estimation
Kdiv = PAR[3]   # Parameter for calculation of K in adaptive neighborhoods

y = np.zeros([Nr,Nc,Np])
K = S**2 - int((S**2)/Kdiv)
print(K)
 
for k in range(Nr):
    leyend = 'Estimating Transmission: ' + str(int(100*(k+1)/Nr)) + '%' # Estimating Airlight:61%
    print(leyend)
    for l in range(Nc):
        f_w = f_proc[ k:S+k, l:S+l, : ]
        f_v = KNB(f_w, K)
        Fmax = f_v.max()
        Fmin = f_v.min()
        range_fv = Fmax - Fmin
        fv_avg = (Fmin+Fmax)/2.0
        if range_fv < w:
            range_fv = w 
        alpha = range_fv/(j0*A_est)
        t_est[k,l] = (A_est - (alpha*fv_avg + (1-alpha)*Fmin)) / (A_est-alpha*fv_avg)
                
        if t_est[k,l] > 1:
            t_est[k,l] = 1
        if t_est[k,l] < w:
            t_est[k,l] = w
print('Done!')

#trans = denoise_tv_chambolle(t_est, weight=1000.0, multichannel=False)
trans = (guidedFilter(np.uint8(f_c),np.uint8(255*t_est),30,0.1) / 255.0) 

y[:,:,0] = (f_c[:,:,0] - A[0]) / (trans +0.00001) + A[0]
y[:,:,1] = (f_c[:,:,1] - A[1]) / (trans +0.00001) + A[1] 
y[:,:,2] = (f_c[:,:,2] - A[2]) / (trans +0.00001) + A[2]
y[:,:,:] = abs(y[:,:,:]) # abs = 絶対値を返す
xs,ys,zs = np.where( y > 255 )
y[xs,ys,zs] = 255.0
                    
#plt.subplot(121), plt.imshow(f_c/255.), plt.title('Hazy Image')
#plt.subplot(122), plt.imshow(y/255.), plt.title('Processed Image')



imsave('D1.png', np.uint8(y)) # 画像の保存






