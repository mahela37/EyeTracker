import cv2
import numpy

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade=cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')



def getEyes(gray_picture):
    eyes = eye_cascade.detectMultiScale(gray_picture, 1.3, 5)
    return eyes

def getMouth(gray_picture):
    mouth_rects = mouth_cascade.detectMultiScale(gray_picture, 1.7, 11)
    return mouth_rects

def getFace(gray_picture):
    face=face_cascade.detectMultiScale(gray_picture,1.3,5)
    return face

def drawRect(image,coords,color,thickness):
    cv2.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]),(color[0], color[1], color[2]), thickness)

def drawRectMouth(image,coords,color,thickness):
    cv2.rectangle(image, (coords[0][0], coords[0][1]), (coords[0][0] + coords[0][2], coords[0][1] + coords[0][3]),(color[0], color[1], color[2]), thickness)


def a_inside_b(x,y):
    return (x[0] > y[0] and x[1] > y[1] and x[0] + x[2] < y[0] + y[2] and x[1] + x[3] < y[1] + y[3])

def a_intersect_b_yAxis(x,y):
    return (x[1]<(y[1]+y[3]))


def processFace(face,eyes,mouth): #do post-processing to remove eye and face rectangles that don't make sense

    #there can only be one face, so remove the others. keep the biggest one
    biggest_face=[0,0,0,0]
    for item in face:
        if(item[2]>biggest_face[2]):
            biggest_face=item

    eyelist=[]

    face_top_half=[biggest_face[0],biggest_face[1],biggest_face[2],biggest_face[3]/2,]
    face_bottom_half = [biggest_face[0], biggest_face[1]+biggest_face[3] / 1.5, biggest_face[2], biggest_face[3] / 2, ]
    #1. checking if the eye rectangle is within the face frame boundaries. get rid of the other eyes.
    #2. checking if the eye is within the upper half of the face
    for eye in eyes:
        if(a_inside_b(eye,biggest_face) and a_intersect_b_yAxis(eye,face_top_half)):
            eyelist.append(eye)

    #if theres a rectangle inside another rectangle, only keep the smaller one.
    eyelist_2 = []
    for eye in eyelist:
        for eye_2 in eyelist:
            if (a_inside_b(eye,eye_2)):
                pass
            else:
                eyelist_2.append(eye_2)


    #mouth filtering
    if(len(mouth)!=0):
        if(a_inside_b(mouth[0],face_bottom_half)):
            pass
        else:
            mouth=[[0,0,0,0]]

    return biggest_face,eyelist_2,mouth

# fileURL="SampleImages/me.jpg"
# img = cv2.imread(fileURL)

#cap=cv2.VideoCapture("SampleImages/Trudeau.mp4")
cap=cv2.VideoCapture("SampleImages/news.mp4")
#cap=cv2.VideoCapture("SampleImages/tech.mp4")

#cap=cv2.VideoCapture(0)    for a webcam
while True:
    _,img=cap.read()
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # make picture gray
    #img = cv2.imread("SampleImages/me.jpg")
    eyes = getEyes(gray_picture)
    face=getFace(gray_picture)
    mouth=getMouth(gray_picture)

    face,eyes,mouth=processFace(face=face,eyes=eyes,mouth=mouth)

    for item in eyes:
        drawRect(img, item, (0, 255, 0), 2)

    drawRect(img, face, (255, 0, 0), 2)

    for item in mouth:
        drawRectMouth(img,mouth,(0,0,255),2)


    cv2.imshow('my image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()