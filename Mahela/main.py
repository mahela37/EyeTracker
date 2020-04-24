import cv2
import numpy

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread("LinkedInHeadShot.png")
gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#make picture gray
faces = eye_cascade.detectMultiScale(gray_picture, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

cv2.imshow('my image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()