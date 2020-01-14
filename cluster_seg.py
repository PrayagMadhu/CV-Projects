import cv2
import numpy
from sklearn.cluster import KMeans


img=cv2.imread("check.png")/255
img_=img.reshape(img.shape[0]*img.shape[1], img.shape[2])

kmeans=KMeans(n_clusters=5).fit(img_)

clustered=kmeans.cluster_centers_[kmeans.labels_]

final=clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
print(final)

#cv2.imshow("df", final)
#cv2.waitKey(0)
