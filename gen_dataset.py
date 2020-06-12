import cv2
import glob
import random
import imutils
import numpy as np
from tqdm import tqdm
import os.path as osp

def gen_image(character, angle, shift, scale):
    rows = character.shape[0]
    cols = character.shape[1]
    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, angle, 1)
    rotated_image = cv2.warpAffine(character, M, (cols, rows), borderValue=(255,255,255))

    dim = (int(rows * scale), int(rows * scale))
    rotated_image = cv2.resize(rotated_image, dim, interpolation = cv2.INTER_AREA)
    rows = rotated_image.shape[0]
    cols = rotated_image.shape[1]

    # color = [0, 0, 0]
    # top, bottom, left, right = [2]*4
    # rotated_image = cv2.copyMakeBorder(rotated_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # rows = rotated_image.shape[0]
    # cols = rotated_image.shape[1]

    img = cv2.imread('grid.png', 1)
    dim = (128, 128)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img[:, :, :] = 255
    start = (dim[0] - rows) // 2
    end = start + rows

    x_min = start+shift[0] + 6
    x_max = end+shift[0] + 6
    y_min = start+shift[1] + 6
    y_max = end+shift[1] + 6

    img[start+shift[0]:end+shift[0], start+shift[1]:end+shift[1], :] = rotated_image

    color = [0, 0, 0]
    top, bottom, left, right = [6]*4
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (x_min, x_max, y_min, y_max)

root = './handwriting'
source = glob.glob(osp.join(root, '*.jpg'))

normal_rotation = [0, -10, 10, -5, 5]
normal_rotation = [-20]
wrong_rotation = [20, -20, -30, 30]

normal_scale = [1, 0.9]
# wrong_scale = [0.8, 0.7, 0.6, 0.5]
wrong_scale = [0.7]

train_images = []
train_bbox = []
train_label = []
# for i in tqdm(range(2000)):
#     path = random.choice(source)
#     img = cv2.imread(path, 1)
#     dim = (90, 90)
#     img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#     img, bbox = gen_image(img, random.choice(normal_rotation), (0, 0), random.choice(normal_scale))

#     # cv2.rectangle(img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0, 0, 0), 3)
#     # cv2.imshow('image', img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     # exit()

#     train_images.append(img)
#     train_bbox.append(bbox)
#     train_label.append(0)

for i in tqdm(range(2000)):
    path = random.choice(source)
    img = cv2.imread(path, 1)
    dim = (90, 90)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img, bbox = gen_image(img, random.choice(normal_rotation), (0, 0), random.choice(wrong_scale))

    cv2.imwrite('./example/ex2.jpg', img)
    exit()
    cv2.rectangle(img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0, 0, 0), 2)
    cv2.imwrite('./example/return.jpg', img)
    print((bbox[0], bbox[1], bbox[2], bbox[3]))
    exit()
    
    train_images.append(img)
    train_bbox.append(bbox)
    train_label.append(1)

for i in tqdm(range(2000)):
    path = random.choice(source)
    img = cv2.imread(path, 1)
    dim = (90, 90)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img, bbox = gen_image(img, random.choice(wrong_rotation), (0, 0), random.choice(normal_scale))
    
    train_images.append(img)
    train_bbox.append(bbox)
    train_label.append(2)

for i in tqdm(range(2000)):
    path = random.choice(source)
    img = cv2.imread(path, 1)
    dim = (90, 90)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img, bbox = gen_image(img, random.choice(wrong_rotation), (0, 0), random.choice(wrong_scale))
    
    train_images.append(img)
    train_bbox.append(bbox)
    train_label.append(3)

train_images = np.array(train_images)
train_bbox = np.array(train_bbox)
train_label = np.array(train_label)
np.save('./dataset/train_images.npy', train_images)
np.save('./dataset/train_bbox.npy', train_bbox)
np.save('./dataset/train_label.npy', train_label)

################################################################################################################################################

train_images = []
train_bbox = []
train_label = []
for i in tqdm(range(500)):
    path = random.choice(source)
    img = cv2.imread(path, 1)
    dim = (90, 90)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img, bbox = gen_image(img, random.choice(normal_rotation), (0, 0), random.choice(normal_scale))

    # cv2.rectangle(img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0, 0, 0), 3)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    train_images.append(img)
    train_bbox.append(bbox)
    train_label.append(0)

for i in tqdm(range(500)):
    path = random.choice(source)
    img = cv2.imread(path, 1)
    dim = (90, 90)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img, bbox = gen_image(img, random.choice(normal_rotation), (0, 0), random.choice(wrong_scale))
    
    train_images.append(img)
    train_bbox.append(bbox)
    train_label.append(1)

for i in tqdm(range(500)):
    path = random.choice(source)
    img = cv2.imread(path, 1)
    dim = (90, 90)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img, bbox = gen_image(img, random.choice(wrong_rotation), (0, 0), random.choice(normal_scale))
    
    train_images.append(img)
    train_bbox.append(bbox)
    train_label.append(2)

for i in tqdm(range(500)):
    path = random.choice(source)
    img = cv2.imread(path, 1)
    dim = (90, 90)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img, bbox = gen_image(img, random.choice(wrong_rotation), (0, 0), random.choice(wrong_scale))
    
    train_images.append(img)
    train_bbox.append(bbox)
    train_label.append(3)

train_images = np.array(train_images)
train_bbox = np.array(train_bbox)
train_label = np.array(train_label)
np.save('./dataset/test_images.npy', train_images)
np.save('./dataset/test_bbox.npy', train_bbox)
np.save('./dataset/test_label.npy', train_label)