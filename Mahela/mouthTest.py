import cv2





def getMouth(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    return mouth_rects


fileURL="SampleImages/reporter.png"
img = cv2.imread(fileURL)
frame=img
cap=cv2.VideoCapture("SampleImages/tech.mp4")

while True:
    _,img=cap.read()
    mouth=getMouth(img)

    for (x,y,w,h) in mouth:
        y = int(y - 0.15*h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        break

    cv2.imshow('Mouth Detector', frame)


cv2.waitKey(0)
cv2.destroyAllWindows()