#find contour and draw it with canny edge detection
import cv2 as cv
import numpy as np
import math

img = cv.imread("tugas3.jpg")

#convert to grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)

#find canny edge
canny = cv.Canny(img_gray, 10, 20)

#find contours
contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    # Contour top left position
    x1,y1 = cnt[0][0]
    # Poly Approximation finding 4 points in border
    approx = cv.approxPolyDP(cnt,0.025*cv.arcLength(cnt,True),True)
    #to find 12 sides of lander
    if len(approx) == 12:
        #to count position and widtheigth 
        x, y, w, h = cv.boundingRect(cnt) 
        

        #draw contours
        cv.drawContours(img, [cnt], -1, (0, 0, 255), 3)

        cv.putText(img, '%i' % len(approx), (40, 100), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4)


            
cv.imshow("Edited",img)
cv.waitKey(0)


cv.imwrite("tugas3_done.jpg", img)

cv.destroyAllWindows()