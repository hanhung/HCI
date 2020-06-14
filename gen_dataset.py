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

    dim = (int(rows * scale), int(cols * scale))
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

    x_min = start+shift[0]+6
    x_max = end+shift[0]+6
    y_min = start+shift[1]+6
    y_max = end+shift[1]+6

    img[start+shift[0]:end+shift[0], start+shift[1]:end+shift[1], :] = rotated_image

    color = [0, 0, 0]
    top, bottom, left, right = [6]*4
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (x_min, x_max, y_min, y_max)

root = './handwriting'
source = glob.glob(osp.join(root, '*.jpg'))

normal_rotation = [0, -10, 10, -5, 5]
wrong_rotation = [20, -20, -30, 30]

normal_scale = [0.88]
wrong_scale = [0.8, 0.7, 0.6, 0.5]

train_images = []
train_bbox = []
train_label = []
train_samples = 50000
test_samples = 10000

for i in tqdm(range(4)):
    path = random.choice(source)
    for j in range(4):
        img = cv2.imread(path, 1)
        dim = (90, 90)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        rotation_choice = np.random.choice(2, 1)[0]
        if rotation_choice == 0:
            rotation_angle = random.choice((-1, 1)) * np.random.randint(low=0, high=11, size=1)[0]
        else:
            rotation_angle = random.choice((-1, 1)) * np.random.randint(low=20, high=41, size=1)[0]

        scale_choice = np.random.choice(2, 1)[0]
        if scale_choice == 0:
            scale_percentage = np.random.uniform(0.9, 1.0, 1)[0]
        else:
            scale_percentage = np.random.uniform(0.5, 0.7, 1)[0]

        shift_choice = np.random.choice(2, 1)[0]
        shift_x = 0
        shift_y = 0
        if shift_choice == 0:
            shift_x = random.choice((-1, 1)) * np.random.randint(low=0, high=4, size=1)[0]
            shift_y = random.choice((-1, 1)) * np.random.randint(low=0, high=4, size=1)[0]
        else:
            shift_x = random.choice((-1, 1)) * np.random.randint(low=6, high=11, size=1)[0]
            shift_y = random.choice((-1, 1)) * np.random.randint(low=6, high=11, size=1)[0]

        img, bbox = gen_image(img, rotation_angle, (shift_x, shift_y), scale_percentage)
        cv2.imwrite('./example/{}.jpg'.format(j * 4 + i), img)
exit()

for i in tqdm(range(train_samples)):
    path = random.choice(source)
    img = cv2.imread(path, 1)
    dim = (90, 90)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    rotation_choice = np.random.choice(2, 1)[0]
    if rotation_choice == 0:
        rotation_angle = random.choice((-1, 1)) * np.random.randint(low=0, high=11, size=1)[0]
    else:
        rotation_angle = random.choice((-1, 1)) * np.random.randint(low=20, high=41, size=1)[0]
    # print('Rotation')
    # print(rotation_choice)
    # print(rotation_angle)

    scale_choice = np.random.choice(2, 1)[0]
    if scale_choice == 0:
        scale_percentage = np.random.uniform(0.9, 1.0, 1)[0]
    else:
        scale_percentage = np.random.uniform(0.5, 0.7, 1)[0]
    # print('Scale')
    # print(scale_choice)
    # print(scale_percentage)

    shift_choice = np.random.choice(2, 1)[0]
    shift_x = 0
    shift_y = 0
    if shift_choice == 0:
        shift_x = random.choice((-1, 1)) * np.random.randint(low=0, high=4, size=1)[0]
        shift_y = random.choice((-1, 1)) * np.random.randint(low=0, high=4, size=1)[0]
    else:
        shift_x = random.choice((-1, 1)) * np.random.randint(low=6, high=11, size=1)[0]
        shift_y = random.choice((-1, 1)) * np.random.randint(low=6, high=11, size=1)[0]
    # print('Shift')
    # print(shift_choice)
    # print(shift_x)
    # print(shift_y)

    img, bbox = gen_image(img, rotation_angle, (shift_x, shift_y), scale_percentage)
    # cv2.rectangle(img, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (255, 0, 0), 1)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print([rotation_choice, scale_choice, shift_choice])
    # exit()

    train_images.append(img)
    train_bbox.append(bbox)
    train_label.append([rotation_choice, scale_choice, shift_choice])

train_images = np.array(train_images)
train_bbox = np.array(train_bbox)
train_label = np.array(train_label)
np.save('./dataset/train_images.npy', train_images)
np.save('./dataset/train_bbox.npy', train_bbox)
np.save('./dataset/train_label.npy', train_label)
print('Train Shapes')
print(train_images.shape)
print(train_bbox.shape)
print(train_label.shape)

################################################################################################################################################

train_images = []
train_bbox = []
train_label = []
for i in tqdm(range(test_samples)):
    path = random.choice(source)
    img = cv2.imread(path, 1)
    dim = (90, 90)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    rotation_choice = np.random.choice(2, 1)[0]
    if rotation_choice == 0:
        rotation_angle = random.choice((-1, 1)) * np.random.randint(low=0, high=11, size=1)[0]
    else:
        rotation_angle = random.choice((-1, 1)) * np.random.randint(low=20, high=41, size=1)[0]
    # print('Rotation')
    # print(rotation_choice)
    # print(rotation_angle)

    scale_choice = np.random.choice(2, 1)[0]
    if scale_choice == 0:
        scale_percentage = np.random.uniform(0.9, 1.0, 1)[0]
    else:
        scale_percentage = np.random.uniform(0.5, 0.7, 1)[0]
    # print('Scale')
    # print(scale_choice)
    # print(scale_percentage)

    shift_choice = np.random.choice(2, 1)[0]
    shift_x = 0
    shift_y = 0
    if shift_choice == 0:
        shift_x = random.choice((-1, 1)) * np.random.randint(low=0, high=4, size=1)[0]
        shift_y = random.choice((-1, 1)) * np.random.randint(low=0, high=4, size=1)[0]
    else:
        shift_x = random.choice((-1, 1)) * np.random.randint(low=6, high=11, size=1)[0]
        shift_y = random.choice((-1, 1)) * np.random.randint(low=6, high=11, size=1)[0]
    # print('Shift')
    # print(shift_choice)
    # print(shift_x)
    # print(shift_y)

    img, bbox = gen_image(img, rotation_angle, (shift_x, shift_y), scale_percentage)
    # cv2.rectangle(img, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (255, 0, 0), 1)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print([rotation_choice, scale_choice, shift_choice])
    # exit()

    train_images.append(img)
    train_bbox.append(bbox)
    train_label.append([rotation_choice, scale_choice, shift_choice])

train_images = np.array(train_images)
train_bbox = np.array(train_bbox)
train_label = np.array(train_label)
np.save('./dataset/test_images.npy', train_images)
np.save('./dataset/test_bbox.npy', train_bbox)
np.save('./dataset/test_label.npy', train_label)
print('Test Shapes')
print(train_images.shape)
print(train_bbox.shape)
print(train_label.shape)
