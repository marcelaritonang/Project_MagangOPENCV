#find contour and draw it with canny edge detection
import cv2 as cv
import numpy as np
import math

img = cv.imread("Tugas2.jpg")

#convert to grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)

# for x in range(0, 250, 5):
#     mask = cv.inRange(img_gray, x, x + 5)
#     print(x)

#     cv.imshow(' ', mask)
#     cv.waitKey(0)

mask = cv.inRange(img_gray, 155, 160)

#find canny edge
canny = cv.Canny(mask, 10, 20)

#find contours
contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    # Contour top left position
    x1,y1 = cnt[0][0]
    # Epsilon
    epsilon = 0.1*cv.arcLength(cnt,True)
    # Poly Approximation finding 4 points in border
    approx = cv.approxPolyDP(cnt,epsilon,True)

    if len(approx) == 4:
        x, y, w, h = cv.boundingRect(cnt)

        if math.isclose(w, h, rel_tol=0.1):
            #draw contours
            cv.drawContours(img, [cnt], -1, (0, 0, 0), 2)

            center_x = x + int(w/2)
            center_y = y + int(h/2)

            cv.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
            cv.putText(img, 'center: x=%i y=%i' % (center_x, center_y), (center_x, center_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


            
cv.imshow("Edited",img)
cv.waitKey(0)


cv.imwrite("Tugas2_done.jpg", img)

cv.destroyAllWindows()
