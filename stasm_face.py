# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 09:55:47 2021

@author: DI
"""
import cv2
import dlib
import matplotlib.pyplot as plt
import sys

img = cv2.imread('C:/Users/DI/Desktop/work/images.jpg', cv2.IMREAD_UNCHANGED)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = hog_face_detector(gray)
for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        with open('pts.txt', 'w') as f:

         for n in range(0,68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(img, (x, y), 1, (0, 255, 255), 1)
            print(y,x , file=f)
            
            plt.imshow(img)
        
filename = 'savedImage.jpg'
cv2.imwrite(filename, img)
# #     cv2.imshow("Face Landmarks", frame)

# #     key = cv2.waitKey(1)
# #     if key == 27:
# #         break
# # cap.release()
# # cv2.destroyAllWindows()
