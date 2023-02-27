import cv2 as cv
import numpy as np
#input gambar original
img = cv.imread("Desktop/RiancoMarcellinoAndreas_5024211061_TugasOpenCV/Tugas1/tugas1.jpg")
overlay = img.copy()

#resize gambar dari 
(h,w) = img.shape[:2]
centerX,centerY= (w //2 ), (h // 2)

# variabel = argumen [y1;y2 , x1:x2]
# topleft  = img[0:centerY, 0:centerX]
# topright  = img[0:centerY, centerX:w]
# bottoleft  = img[centerY:h, 0:centerX]
# bottomright  = img[centerY:h, centerX:w]

cv.rectangle(img,(0,0),(int(centerX),int(centerY)),(100,100,0),-1)
cv.rectangle(img,(int(centerX),0),(int(w),int(centerX)),(0,100,0),-1)
cv.rectangle(img,(0,int(centerY)),(int(centerX),int(h)),(100,0,100),-1)
cv.rectangle(img,(int(centerX),int(centerY)),(int(w),int(h)),(0,50,50),-1)


alpha=0.4
img_new = cv.addWeighted(img,alpha, overlay, alpha,1,overlay)
cv.imshow("split colour",img_new )  
cv.waitKey(0)
cv.destroyAllWindows()
