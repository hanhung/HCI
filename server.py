import cv2
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model import *

import werkzeug, os
from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)
UPLOAD_FOLDER = './example'
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')

model = Net()

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class PhotoUpload(Resource):
    decorators=[]

    def post(self):
        data = parser.parse_args()
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }
        photo = data['file']

        if photo:
            filename = 'received.jpg'
            path = os.path.join(UPLOAD_FOLDER,filename)
            photo.save(path)

            grid_img = cv2.imread(path, 1)
            grid_img = grid_img / 255

            return_bbox = []
            return_label = []
            
            rows = []
            for i in range(4):
                cols = []
                for j in range(4):
                    img = torch.from_numpy(grid_img[i*140:i*140+140, j*140:j*140+140]).float().permute(2, 0, 1).unsqueeze(0)
                    pred_bbox, pred_label_1, pred_label_2, pred_label_3 = model(img)
                    img = img[0].permute(1, 2, 0).detach().cpu().numpy()
                    box = pred_bbox[0].detach().cpu().numpy() * img.shape[0]
                    pred_label_1 = pred_label_1.argmax(-1)[0]
                    pred_label_2 = pred_label_2.argmax(-1)[0]
                    pred_label_3 = pred_label_3.argmax(-1)[0]

                    return_bbox.append([box[1], box[0], box[3] + box[1], box[2] + box[0]])
                    return_label.append([pred_label_1.item(), pred_label_2.item(), pred_label_3.item()])
                    print(return_bbox[-1])
                    print(return_label[-1])

                    img = img * 255
                    img = cv2.rectangle(img, (box[1], box[0]), (box[3] + box[1], box[2] + box[0]), (0, 0, 255), 1)

                    outstr = ''
                    if pred_label_1 == 1:
                        outstr += 'Rotated'
                        if pred_label_2 == 1:
                            outstr += ', '
                        elif pred_label_3 == 1:
                            outstr += ', '
                    if pred_label_2 == 1:
                        outstr += 'Scaled'
                        if pred_label_3 == 1:
                            outstr += ', '
                    if pred_label_3 == 1:
                        outstr += 'Shifted '
                    if pred_label_1 == 0 and pred_label_2 == 0 and pred_label_3 == 0:
                        outstr = 'Normal'

                    img = cv2.putText(img, outstr, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.imwrite('./example/test{}.jpg'.format(i*4+j), img)
                    cols.append(img)
                im_h = cv2.hconcat(cols)
                rows.append(im_h)
            im_v = cv2.vconcat(rows)
            cv2.imwrite('./example/predicted.jpg', im_v)
            return_bbox = np.array(return_bbox)
            return_label = np.array(return_label)
            np.save('./example/return_bbox.npy', return_bbox)
            np.save('./example/return_label.npy', return_label)
            # print(return_bbox.shape)
            # print(return_label.shape)
            # time.sleep(1000)

            return {
                    'bbox': return_bbox.tolist(),
                    'label': return_label.tolist(),
                    'message': 'photo uploaded',
                    'status': 'success'
                    }
        return {
                'data': '',
                'message': 'Something when wrong',
                'status': 'error'
                }

api.add_resource(HelloWorld, '/')
api.add_resource(PhotoUpload,'/upload')

if __name__ == '__main__':
    model.load_state_dict(torch.load('model.t7', map_location=torch.device('cpu')))
    app.run(debug=True)