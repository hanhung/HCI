import glob
import os.path as osp
import pickle
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
 
 
root = './character'
dataset = glob.glob(osp.join(root, '*_vectors'))
 
save_dir = './handwriting'

root = './character'
label_path = osp.join(root, 'labels.txt')
char_dict = {}
# 此处一定要加上encoding='gbk',否则默认utf-8解析会出错
with open(label_path, 'r', encoding='gbk') as f:
    label_str = f.read()
    for i in range(len(label_str)):
        char_dict[i] = label_str[i]
# 可以看到一共有labels有6825个字符，与解析vectors文件得到的字符数是相同的
print("number of char: ", len(char_dict))
with open('char_dict_HIT-OR3C', 'wb') as f:
    pickle.dump(char_dict, f)
 
def drawStroke(img, pts):
    length = len(pts)
    for i in range(1, length):
        # color和thickness可以根据需要自己更改
        cv2.line(img, (pts[i-1][0], pts[i-1][1]), (pts[i][0], pts[i][1]), color=(0, 0, 0), thickness=3)
    return img
 
for file_id in tqdm(range(len(dataset))):
    path = dataset[file_id]
    with open(path, 'rb') as f:
        # 读取vectors文件包含的字符个数，该值占4个字节，按照uint32读取
        # 此处可知vectors文件读取的字符个数为6825与label的个数相对应
        sample_num = np.fromfile(f, dtype='uint32', count=1)[0]
        
        # 接下来是读取描述每个字符所需要的字节数，该值占2个字节，按照uint16读取
        samples_byte = []
        for sample_id in range(sample_num):
            sample_byte = np.fromfile(f, dtype='uint16', count=1)[0]
            samples_byte.append(sample_byte)
        
        # 接下来从每个字符的描述中重构出该字符
        for sample_id in range(sample_num):
            # init image canva
            img = np.ones((128, 128), dtype=np.uint8) * 255
            counter = 0
            
            # 读取该字符的笔画数，该值占1个字节，按照uint8读取
            stroke_num = np.fromfile(f, dtype='uint8', count=1)[0]
            counter += 1
            
            # 按顺序读取每笔的采样点个数，该值占1个字节，按照uint8读取
            # 就是这里不能按照File Format Specification文件里说明的signed char格式即int8形式读取
            # 否则存在有些笔画采样了128个点，若按照int8读取，会得到-128的结果，造成后续解析出错。
            strokes_points_num = []
            for stroke_id in range(stroke_num):
                stroke_points_num = np.fromfile(f, dtype='uint8', count=1)[0]
                strokes_points_num.append(stroke_points_num)
                counter += 1
 
            # 接下来是按照顺序读取每笔的采样点，一对坐标(x,y)占2个字节，按照uint8读取
            for num in strokes_points_num:
                points = np.fromfile(f, dtype='uint8', count=num*2).reshape((-1, 2))
                counter += num*2
                # 由于记录的是采样点，所以需要连接相邻点把这一笔画出来
                img = drawStroke(img, points)
            
            # 最后判断下描述该字符所用的字节数是否和最开始解析得到的值相同，也算是验证下解析过程是否正确
            assert counter == samples_byte[sample_id], "Error! counter: {}, sample byte: {}".format(counter, samples_byte[sample_id])
            
            # 最后将重构的字符保存下来，以label命名
            label = char_dict[sample_id]
            save_name = label + '_' + str(file_id) + '.jpg'
            image = Image.fromarray(img)
            image.save(osp.join(save_dir, save_name))