import cv2
import numpy
from imutils import face_utils

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade=cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    #taken from: https://gist.github.com/clungzta/b4bbb3e2aa0490b0cfcbc042184b0b4e
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """
    img_to_overlay_t=cv2.imread(img_to_overlay_t,-1)


    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img

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

import dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#cap=cv2.VideoCapture("SampleImages/Trudeau.mp4")
cap=cv2.VideoCapture("SampleImages/me.mp4")
#cap=cv2.VideoCapture("SampleImages/tech.mp4")
#cap=cv2.VideoCapture("SampleImages/news.mp4")

#cap=cv2.VideoCapture(0)    for a webcam
while True:

    _,img=cap.read()
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # make picture gray
    #img = cv2.imread("SampleImages/me.j0pg")
    eyes = getEyes(gray_picture)
    face=getFace(gray_picture)
    mouth=getMouth(gray_picture)

    face,eyes,mouth=processFace(face=face,eyes=eyes,mouth=mouth)

    for item in eyes:
        drawRect(img, item, (0, 255, 0), 2)

    drawRect(img, face, (255, 0, 0), 2)

    for item in mouth:
        drawRectMouth(img,mouth,(0,0,255),2)

    #beardPath='SampleImages/beard.png'
    #TODO: need to make the dimensions and position offset dynamic based on face rectangle
    #img=overlay_transparent(img,beardPath,face[0]-30, face[1]-20, (280, 300))

    #sunglassesPath='SampleImages/sunglasses.png'
    #img = overlay_transparent(img, sunglassesPath, face[0] , face[1]-15, (200, 200))

    #Labels
    # cv2.putText(img, "Face", (face[0], face[1]), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1)
    # cv2.putText(img, "Mouth", (mouth[0][0], mouth[0][1]), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1)
    # cv2.putText(img, "Eye", (eyes[0][0], eyes[0][1]), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1)
    # cv2.putText(img, "Eye", (eyes[1][0], eyes[1][1]), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1)

    #now let's extract 68 points of interest from the face. using dlib for this
    rect=dlib.rectangle(face[0],face[1],face[0]+face[2],face[1]+face[3])
    shape = predictor(gray_picture, rect)
    shape = face_utils.shape_to_np(shape)

    for (x, y) in shape:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('my image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()