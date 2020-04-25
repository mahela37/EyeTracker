import cv2
import numpy

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getEyes(img,gray_picture):
    eyes = eye_cascade.detectMultiScale(gray_picture, 1.3, 5)
    return eyes

def getFace(img,gray_picture):
    face=face_cascade.detectMultiScale(gray_picture,1.3,5)
    return face

def drawRect(image,coords,color,thickness):
    cv2.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]),(color[0], color[1], color[2]), thickness)


def processFace(face,eyes): #do post-processing to remove eye and face rectangles that don't make sense

    #there can only be one face, so remove the others. keep the biggest one
    biggest_face=[0,0,0,0]
    for item in face:
        if(item[2]>biggest_face[2]):
            biggest_face=item

    eyelist=[]

    #basically checking if the eye rectangle is within the face frame boundaries. get rid of the other eyes.
    for eye in eyes:
        if(eye[0]>biggest_face[0] and eye[1]>biggest_face[1] and eye[0]+eye[2]<biggest_face[0]+biggest_face[2] and eye[1]+eye[3]<biggest_face[1]+biggest_face[3]):
            eyelist.append(eye)


    return biggest_face,eyelist

# fileURL="SampleImages/me.jpg"
# img = cv2.imread(fileURL)

#cap=cv2.VideoCapture("SampleImages/Trudeau.mp4")
cap=cv2.VideoCapture("SampleImages/tech.mp4")

#cap=cv2.VideoCapture(0)    for a webcam
while True:
    _,img=cap.read()
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # make picture gray
    #img = cv2.imread("SampleImages/me.jpg")
    eyes = getEyes(img,gray_picture)
    face=getFace(img,gray_picture)

    face,eyes=processFace(face=face,eyes=eyes)
    for item in eyes:
        drawRect(img, item, (0, 255, 0), 2)

    drawRect(img, face, (255, 0, 0), 2)

    cv2.imshow('my image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()