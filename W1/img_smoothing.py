import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('example.png')
assert img is not None, "file could not be read, check with os.path.exists()"

meanBlur = cv2.blur(img, (11, 11))
medianBlur = cv2.medianBlur(img, 11)
gaussBlur = cv2.GaussianBlur(img, (11, 11), 0)
biBlur = cv2.bilateralFilter(img, 15, 75, 75)

plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(meanBlur),plt.title('Mean blur')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(gaussBlur),plt.title('Gaussian blur')
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(medianBlur),plt.title('Median blur')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(biBlur),plt.title('Bilateral blur')
plt.xticks([]), plt.yticks([])
plt.show()