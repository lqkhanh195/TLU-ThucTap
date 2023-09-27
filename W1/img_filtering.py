import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

kernel = np.ones((5,5),np.float32)/10
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Filtered')
plt.xticks([]), plt.yticks([])
plt.show()
