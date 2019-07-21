import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

ein_img = cv2.imread('einstein.png',0).astype(np.float32) / 255.
mar_img = cv2.imread('marilyn.png',0).astype(np.float32) / 255

blurred = cv2.GaussianBlur(mar_img, (5,5), 15) 
blurred_fft = np.fft.fftshift(np.fft.fft2(blurred, norm=None)).astype(np.float32)

def direct_blur():

    low_pass_mar = cv2.GaussianBlur(mar_img, (9,9), 15)
    low_pass_ein = cv2.GaussianBlur(ein_img, (9,9), 15)
    high_pass_ein = ein_img - low_pass_ein

    res = high_pass_ein + low_pass_mar

    cv2.imwrite('direct.png', res * 255)

def gaussian(x, y, a, b, sig):
    return math.e ** (-((x-a)**2+(y-b)**2)/(2*(sig**2)))

def fourier_blur():
    f = np.fft.fft2(mar_img, norm=None)
    f_shift = np.fft.fftshift(f)
    #res = f.astype(np.float32)
    kernel = np.zeros(f.shape).astype(np.float32)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            kernel[i, j] = gaussian(i, j, f.shape[0]/2, f.shape[1]/2, 15)

    kernel = kernel.astype(np.float32) / kernel.sum() 
    f_kernel = np.fft.fft2(kernel, norm=None)
    f_kernel_shift = np.fft.fftshift(f_kernel)

    f_res = f_shift * f_kernel_shift
    res = f_res.astype(np.float32)
    # res = np.fft.ifft2(np.fft.ifftshift(f_res), norm='ortho')
    # res = np.real(res)
    # mag = 20*np.log(np.abs(f_kernel))
    print(np.mean(blurred_fft / res))
    plt.subplot(1, 2, 1)
    plt.imshow(res, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(blurred_fft, cmap='gray')
    plt.show()

#direct_blur()
fourier_blur()
