import cv2
from skimage.filters import threshold_local
import numpy as np
import argparse
import imutils
from pyimagesearch.transform import four_point_transform

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path for image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
ratio=image.shape[0]/500.0
orig=image.copy()
image=imutils.resize(image, height=500)

gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray, (5,5), 0)
edged=cv2.Canny(blur, 95, 250)

cnts=cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
	per=cv2.arcLength(c, True)
	apprx=cv2.approxPolyDP(c, 0.02*per, True)
	if len(apprx)==4:
		bound=apprx
		break

cropped=four_point_transform(orig, bound.reshape(4,2)*ratio)
cropped=cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
T = threshold_local(cropped, 11, offset = 10, method = "gaussian")
cropped = (cropped > T).astype("uint8") * 255

cv2.imshow("orginal", imutils.resize(orig, height=650))
cv2.imshow("scanned", imutils.resize(cropped, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()
