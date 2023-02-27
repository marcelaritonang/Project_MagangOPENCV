import numpy as np
import cv2
  
# Menangkap video melalui webcam
webcam = cv2.VideoCapture(0)

#perulangan   
while(1):
      
    # Membaca video dari
    # webcam dalam bingkai gambar
    _, imageFrame = webcam.read()
  
    # Convert to hsv
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for yellow color and 
    #  mask
    yellow_lower = np.array([25, 90, 50], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
  
    # Set range for red color and 
    # #  mask
    # red_lower = np.array([0, 170, 59], np.uint8)
    # red_upper = np.array([10, 255, 255], np.uint8)
    # red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    red_lower = np.array([170,20, 70], np.uint8)
    red_upper = np.array([185, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
  
    # Set range for green color and 
    #  mask
    green_lower = np.array([50  , 100, 100], np.uint8)
    green_upper = np.array([70, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
  
    # Set range for blue color and
    #  mask
    blue_lower = np.array([108, 110, 50], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
      
     # untuk setiap operator warna dan bitwise_and
     # antara imageFrame dan mask menentukan
     # untuk mendeteksi hanya warna tertentu itu
    kernal = np.ones((5, 5), "uint8")

    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame, 
                              mask = yellow_mask)
      
    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                              mask = red_mask)
      
    #green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask = green_mask)
      
    #blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = blue_mask)
    
    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h), 
                                       (36, 255, 255), 2)
              
            cv2.putText(imageFrame, "yellow Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (35, 255, 255))    
    
   
    #Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h), 
                                       (0, 0, 255), 2)
              
            cv2.putText(imageFrame, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))    
  
    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                       (x + w, y + h),
                                       (0, 255, 0), 2)
              
            cv2.putText(imageFrame, "Green Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0))
  
    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)
              
            cv2.putText(imageFrame, "Blue Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))
              
    
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break