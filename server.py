import cv2
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
            img = cv2.imread(path, 1)
            img = img / 255
            img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            pred_bbox, pred_label = model(img)

            img = img[0].permute(1, 2, 0).detach().cpu().numpy()
            box = pred_bbox[0].detach().cpu().numpy() * img.shape[0]
            label = pred_label[0].argmax(-1).detach().cpu().numpy()

            fig,ax = plt.subplots(1)
            ax.imshow(img)
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            comment = None
            if (label == 0):
                ax.text(10, 10, "Good Job")
                comment = "Good Job"
            elif (label == 1):
                ax.text(10, 10, "Text Too Small")
                comment = "Text Too Small"
            elif (label == 2):
                ax.text(10, 10, "Text Is Rotated")
                comment = "Text Is Rotated"
            else:
                ax.text(10, 10, "Text Too Small, Text Is Rotated")
                comment = "Text Too Small, Text Is Rotated"
            fig.savefig(os.path.join(UPLOAD_FOLDER,'model.png'), dpi=90, bbox_inches='tight')

            fig,ax = plt.subplots(1)
            ax.imshow(img)
            fig.savefig(os.path.join(UPLOAD_FOLDER,'original.png'), dpi=90, bbox_inches='tight')
            
            outstr = 'Bounding Box: (x_min: {}, y_min: {}, height: {}, width: {}), Comment: {}'.format(box[0], box[1], box[2], box[3], comment)

            return {
                    'data':outstr,
                    'message':'photo uploaded',
                    'status':'success'
                    }
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }


api.add_resource(HelloWorld, '/')
api.add_resource(PhotoUpload,'/upload')

if __name__ == '__main__':
    model.load_state_dict(torch.load('model.t7', map_location=torch.device('cpu')))
    app.run(debug=True)