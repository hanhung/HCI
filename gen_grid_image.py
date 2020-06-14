import cv2
import glob
import random
import imutils
import numpy as np
from tqdm import tqdm
import os.path as osp

# im1 = cv2.imread('./example/0.jpg')
# im2 = cv2.imread('./example/1.jpg')

# im_v = cv2.vconcat([im1, im1])
# cv2.imwrite('./example/opencv_vconcat.jpg', im_v)

# im_h = cv2.hconcat([im1, im1])
# cv2.imwrite('./example/opencv_hconcat.jpg', im_h)

rows = []
for i in range(4):
    im1 = cv2.imread('./example/{}.jpg'.format(i*4+0))
    im2 = cv2.imread('./example/{}.jpg'.format(i*4+1))
    im3 = cv2.imread('./example/{}.jpg'.format(i*4+2))
    im4 = cv2.imread('./example/{}.jpg'.format(i*4+3))
    im_h = cv2.hconcat([im1, im2, im3, im4])
    rows.append(im_h)
im_v = cv2.vconcat(rows)
cv2.imwrite('./example/send.jpg', im_v)
print(im_v.shape)
