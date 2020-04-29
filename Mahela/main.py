import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier('HaarCascades/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('HaarCascades/haarcascade_frontalface_default.xml')
mouth_cascade=cv2.CascadeClassifier('HaarCascades/haarcascade_mcs_mouth.xml')

#Used to convert dlib.rectangle type back to a Numpy array for easier addressing
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

#Overlays one image on top of the other.
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

#Return the eyes as rectangles
def getEyes(gray_picture):
    eyes = eye_cascade.detectMultiScale(gray_picture, 1.3, 5)
    return eyes

#Return the mouths as rectangles
def getMouth(gray_picture):
    mouth_rects = mouth_cascade.detectMultiScale(gray_picture, 1.7, 11)
    return mouth_rects

#Return faces as rectangles
def getFace(gray_picture):
    face=face_cascade.detectMultiScale(gray_picture,1.3,5)
    return face

#Draws a rectangle over an image
def drawRect(image,coords,color,thickness):
    cv2.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]),(color[0], color[1], color[2]), thickness)

#Slighly different for the mouth
def drawRectMouth(image,coords,color,thickness):
    cv2.rectangle(image, (coords[0][0], coords[0][1]), (coords[0][0] + coords[0][2], coords[0][1] + coords[0][3]),(color[0], color[1], color[2]), thickness)

#Checks if rectangle A is completely inside rectangle B
def a_inside_b(a,b):
    return (a[0] > b[0] and a[1] > b[1] and a[0] + a[2] < b[0] + b[2] and a[1] + a[3] < b[1] + b[3])

#Checks if rectangle A's y top is above rectangle B's y bottom
def a_intersect_b_yAxis(a,b):
    return (a[1]<(b[1]+b[3]))


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
    #2. checking if the eye starts within the upper half of the face
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


    #mouth filtering. make sure the mouth is in the bottom half of the face
    if(len(mouth)!=0):
        if(a_inside_b(mouth[0],face_bottom_half)):
            pass
        else:
            mouth=[[0,0,0,0]]

    return biggest_face,eyelist_2,mouth


######################Different files I've used to test the code #############################
# fileURL="SampleImages/me.jpg"
# img = cv2.imread(fileURL)

#cap=cv2.VideoCapture("SampleImages/Trudeau.mp4")
cap=cv2.VideoCapture("SampleImages/me.mp4")
#cap=cv2.VideoCapture("SampleImages/tech.mp4")
#cap=cv2.VideoCapture("SampleImages/news.mp4")
#cap=cv2.VideoCapture(0)    for a webcam

###############################################################################################

import dlib #dlib is used as the machine learning library
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   #we train it using a dataset used to detect 68 facial features

#Plot a range of dlib x,y coordinates on an image. ConnectEnds flag connects the first and last dot to create a closed polygon
def dlibPlotRange(start, end, img, coords, connectEnds=0):
    start_0=start
    while(start<end):
        cv2.line(img, (coords[start][0], coords[start][1]), (coords[start+1][0], coords[start+1][1]), (255, 0, 0), 1)
        start=start+1
    if(connectEnds):
        cv2.line(img, (coords[start_0][0], coords[start_0][1]), (coords[end][0], coords[end][1]), (255, 0, 0),1)

#Plots individual dlib x,y coordinates on an image
def dlibPlotPoints(points,img,coords):
    plen=len(points)
    i=0
    while(i<plen):
        cv2.line(img, (coords[points[i]][0], coords[points[i]][1]), (coords[points[i+1]][0], coords[points[i+1]][1]), (255, 0, 0), 1)
        if(i+2==len(points)):
            break
        i=i+1

#Plots dlib features by feature name
def dlibPlotFeatures(feature,img,coords):
    if(feature=='chin'):
        dlibPlotRange(0, 16, img, coords)

    elif(feature=='left_eyebrow'):
        dlibPlotRange(17, 21, img, coords,connectEnds=1)

    elif (feature=='right_eyebrow'):
        dlibPlotRange(22, 26, img, coords,connectEnds=1)

    elif (feature == 'nose_line'):
        dlibPlotRange(27, 30, img, coords)

    elif (feature=='nostrils'):
        dlibPlotRange(31, 35, img, coords)

    elif (feature=='left_eye'):
        dlibPlotRange(36, 41, img, coords,connectEnds=1)

    elif (feature=='right_eye'):
        dlibPlotRange(42, 47, img, coords,connectEnds=1)

    elif (feature=='lips'):
        dlibPlotRange(48, 67, img, coords)

    elif (feature=='left_eyeball'):
        dlibPlotPoints((37,38,40,41,37), img, coords)

    elif (feature=='right_eyeball'):
        dlibPlotPoints((43,44,46,47,43), img, coords)



while True:

    _,img=cap.read()

    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # make picture gray. makes it faster for opencv and dlib
    eyes = getEyes(gray_picture)
    face=getFace(gray_picture)
    mouth=getMouth(gray_picture)
    face,eyes,mouth=processFace(face=face,eyes=eyes,mouth=mouth)    #do our filtering

    for item in eyes:
        drawRect(img, item, (0, 255, 0), 2)

    drawRect(img, face, (255, 0, 0), 2)

    for item in mouth:
        drawRectMouth(img,mouth,(0,0,255),2)

    ############Overlay code that I was using to draw stuff on top of face ##############
    #beardPath='SampleImages/beard.png'
    #TODO: need to make the dimensions and position offset dynamic based on face rectangle
    #img=overlay_transparent(img,beardPath,face[0]-30, face[1]-20, (280, 300))

    #sunglassesPath='SampleImages/sunglasses.png'
    #img = overlay_transparent(img, sunglassesPath, face[0] , face[1]-15, (200, 200))
    ######################################################################################

    #Labels that categorize each rectangle
    # cv2.putText(img, "Face", (face[0], face[1]), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1)
    # cv2.putText(img, "Mouth", (mouth[0][0], mouth[0][1]), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1)
    # cv2.putText(img, "Eye", (eyes[0][0], eyes[0][1]), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1)
    # cv2.putText(img, "Eye", (eyes[1][0], eyes[1][1]), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 255), 1)

    #now let's extract 68 points of interest from the face. using dlib for this
    rect=dlib.rectangle(face[0],face[1],face[0]+face[2],face[1]+face[3])    #First, create a dlib.rectangle type with the face frame's starting and ending (x,y) coordinates
    shape = predictor(gray_picture, rect)
    shape = shape_to_np(shape)   #Converting the results back from dlib.rectangle type to a Numpy array for easier addressing

    for (x, y) in shape:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)


    dlibPlotFeatures('right_eyebrow',img,shape)
    dlibPlotFeatures('left_eyebrow', img, shape)
    dlibPlotFeatures('nose_line', img, shape)
    dlibPlotFeatures('nostrils', img, shape)
    dlibPlotFeatures('lips', img, shape)
    dlibPlotFeatures('left_eye', img, shape)
    dlibPlotFeatures('left_eyeball', img, shape)
    dlibPlotFeatures('right_eye', img, shape)
    dlibPlotFeatures('right_eyeball', img, shape)
    dlibPlotFeatures('chin', img, shape)

    cv2.imshow('test', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()