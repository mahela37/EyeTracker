
import cv2
import numpy as np

im = cv2.imread("SampleImages/me.jpg",0)

ret,im = cv2.threshold(im,20,255,cv2.THRESH_BINARY_INV)

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 1
params.maxThreshold = 255

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByInertia = True
params.minInertiaRatio = 0.1


params.filterByConvexity = True
params.minConvexity = 0.1

params.filterByArea = True
params.minArea = 40

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)



"""
ret,im = cv2.threshold(im,20,255,cv2.THRESH_BINARY_INV)
from matplotlib import pyplot as plt
plt.imshow(im)
plt.show()


"""




