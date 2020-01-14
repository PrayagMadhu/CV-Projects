import imutils
import cv2

img=cv2.imread("shapes.jpg")

# for finding contour(outline of shape) -> greyscale -> gauss blur -> threshold (converts to binary img(black and white))

gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur=cv2.GaussianBlur(gray, (5,5),0) #( <input img>, (dim of filter kernal), std. dev. from x and y axis 					     if 0 given, then default value taken from img )

thres=cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1] #first o/p is ret value; if the pixel value is 			less than thresold(here, 60), the replace the pixel by given value (here 255)

cntrs=cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] # returns list of array 											   of contours
#cntrs=imutils.grab_contours(cntrs) # from the  above list splits up individual contour arrays

for c in cntrs:
	m=cv2.moments(c) # returns dict of moment values (m00, m01, m10...); 
			# centre is calculated cx=m10/m00; cy=m01/m00
	if m["m00"]==0.0:
		Cx=0
		Cy=0
	else:
	
		Cx=int(m["m10"] / m["m00"])
		Cy=int(m["m01"]/m["m00"])
	
	cv2.drawContours(img, c, -1, (0,255,0), 3)
	cv2.circle(img, (Cx, Cy), 7, (0,255,0), -1)
	cv2.imshow("sample", img)
	cv2.waitKey(0)

