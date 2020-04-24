import cv2
import numpy

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread("LinkedInHeadShot.png")
img = cv2.imread("me.jpg")
gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#make picture gray

eyes = eye_cascade.detectMultiScale(gray_picture, 1.3, 5)


if(eyes[0][0]<eyes[1][0]):
    left_eye = eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3]
    right_eye = eyes[1][0], eyes[1][1], eyes[1][2], eyes[1][3]
else:
    right_eye = eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3]
    left_eye = eyes[1][0], eyes[1][1], eyes[1][2], eyes[1][3]


def drawRect(image,coords,color,thickness):
    cv2.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]),(color[0], color[1], color[2]), thickness)


drawRect(img,left_eye,(255,255,0),2)
drawRect(img,right_eye,(255,255,0),2)

cv2.imshow('my image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()