import cv2
import numpy as np
from skimage.io import imread, imsave



# 処理対象の画像をロード
imgS = cv2.imread("D2.png")

# ガンマ値を決める。
gamma = 1.2


# ガンマ値を使って Look up tableを作成
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

# Look up tableを使って画像の輝度値を変更
imgA = cv2.LUT(imgS, lookUpTable)

# 表示実行
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', imgA)
cv2.waitKey()
imgB = imgA[:,:,::-1]
imsave('G1.png', np.uint8(imgB)) # 画像の保存
