import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

img = Image.open('lena.png').convert('L')
img = np.array(img).astype(np.float32) / 255.
h, w = img.shape[:2]
x_edge = abs(img[:, 1:] - img[:, :w-1])
y_edge = abs(img[1:, :] - img[:h-1, :])
x_edge = x_edge[:-1, :]
y_edge = y_edge[:, :-1]

gradient = np.sqrt((x_edge * x_edge) + (y_edge * y_edge))

print(gradient.max(), gradient.min(), gradient.mean())
t = 0.02
gradient[gradient < t] = 0
img_canny = cv2.Canny(np.uint8(255 * img), 100, 200)

plt.subplot(131)
plt.imshow(np.uint8(x_edge * 255), cmap='gray')
plt.subplot(132)
plt.imshow(np.uint8(img_canny), cmap='gray')
plt.subplot(133)
plt.imshow(np.uint8(gradient * 255), cmap='gray')
plt.show()