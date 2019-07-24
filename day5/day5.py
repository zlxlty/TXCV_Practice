'''
@Description: Edit
@Author: Tianyi Lu
@Date: 2019-07-24 09:56:34
@LastEditors: Tianyi Lu
@LastEditTime: 2019-07-24 11:00:49
'''
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path1 = '1969469165.jpg'
path2 = '1039691984.jpg'

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

'''
Check the difference between different interpolation
'''
# img1_small = cv2.resize(img1, (0, 0), fx=0.125,fy=0.125,
#                         interpolation=cv2.INTER_NEAREST)
# img1_small_linear = cv2.resize(img1, (0, 0), fx=0.125,fy=0.125,
#                         interpolation=cv2.INTER_LINEAR)

img1_small = cv2.resize(img1, (0, 0), fx=0.125,fy=0.125,
                        interpolation=cv2.INTER_AREA)
img2_small = cv2.resize(img2, (0, 0), fx=0.125,fy=0.125,
                        interpolation=cv2.INTER_AREA)

# plt.subplot(221)
# plt.imshow(img1_small)
# plt.subplot(222)
# plt.imshow(img2_small)
# plt.show()

# ORB descriptor
MAX_FEATURE = 200 # look for 50 feature in the graph
orb = cv2.ORB_create(MAX_FEATURE)

keypoint1, descriptor1 = orb.detectAndCompute(img1_small, None)
keypoint2, descriptor2 = orb.detectAndCompute(img2_small, None)
# print('descriptor shape:', descriptor1.shape)

'''
Draw keypoints overlay onto iamge
'''
img_vis1 = cv2.drawKeypoints(img1_small, keypoint1, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img_vis2 = cv2.drawKeypoints(img2_small, keypoint2, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# plt.subplot(223)
# plt.imshow(img_vis1)
# plt.subplot(224)
# plt.imshow(img_vis2)
# plt.show()

'''
Matching descriptor
'''
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)
matches = matcher.match(descriptor1, descriptor2, None)

print("First match: %s and total match number %d" % (matches[0], len(matches)))

matches.sort(key=lambda x: x.distance, reverse=False)
print(matches[0].queryIdx, matches[0].trainIdx, matches[0].distance)

GOOD_MATCH_PERCENT = 0.2
numGoodMatch = int(len(matches) * GOOD_MATCH_PERCENT)
good_matches = matches[:numGoodMatch]
print("Number of good matches: %d"%len(good_matches))

imMatch = cv2.drawMatches(img1_small, keypoint1, img2_small, keypoint2, matches, None)

imGoodMatch = cv2.drawMatches(img1_small, keypoint1, img2_small, keypoint2, good_matches, None)

# plt.subplot(121)
# plt.imshow(imMatch)
# plt.subplot(122)
# plt.imshow(imGoodMatch)
# plt.show()

'''
Extrach matching descriptors coordinates
'''
points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

for i, match in enumerate(good_matches):
    points1[i, :] = keypoint1[match.queryIdx].pt
    points2[i, :] = keypoint2[match.trainIdx].pt

H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, maxIters=30)

print(H)
'''
Use computed homography to warp image
'''
height, width = img1_small.shape[:2]
img1warp = cv2.warpPerspective(img1_small, H, (width, height))

plt.subplot(121)
plt.imshow(img1warp)
plt.subplot(122)
plt.imshow(img2_small)
plt.show()
