import numpy as np 
import cv2 
from matplotlib import pyplot as plt

Img = cv2.imread("Dibujos.jpg",cv2.COLOR_BGR2GRAY)
gris = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

mask = np.zeros(Img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

############################# ROI ##############################

roi = cv2.selectROI(Img)
Segmentada = Img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cv2.imwrite("ROI.jpg",Segmentada)


############################### FONDO ######################################

cv2.grabCut(Img,mask,roi,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
Img = Img*mask2[:,:,np.newaxis]



########################### CORNERS ##################################

cv2.imwrite("Corner.jpg",Img)

temcolor = cv2.imread('Corner.jpg',1)

gray = cv2.cvtColor(temcolor,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)


corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)


for corner in corners:
    x,y = corner.ravel()
    cv2.circle(temcolor,(x,y),3,255,-1)


plt.imshow(Img)
plt.colorbar()
plt.show()

cv2.imshow('Corner',temcolor)
cv2.waitKey(0)



cv2.destrollAllWindows()











