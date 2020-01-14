import cv2
from matplotlib  import pyplot as plt
import numpy as np

img=cv2.imread("check.png")
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gray_r=gray.reshape(gray.shape[0]*gray.shape[1])

for i in range(gray_r.shape[0]):
	 if gray_r[i]<gray_r.mean():
	 	gray_r[i]=255
	 else:
	 	gray_r[i]=0
	 	
gray_r_=gray_r.reshape(gray.shape[0], gray.shape[1])

#print(gray_r_)
cv2.imshow("df", gray_r_)
cv2.waitKey(0)
