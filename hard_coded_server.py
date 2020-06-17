import cv2
import time
import numpy as np

import werkzeug, os
from flask import jsonify
from flask import Flask, request
from flask_ngrok import run_with_ngrok
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)
UPLOAD_FOLDER = './example'
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class PhotoUpload(Resource):
    decorators=[]

    def post(self):
        data = request.files['file'].read()
        npimg = np.fromstring(data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img.shape[0] == 560:
            return_bbox = np.load('./example/return_bbox.npy')
            return_label = np.load('./example/return_label.npy')

            response = jsonify({
                    'bbox': return_bbox.tolist(),
                    'label': return_label.tolist(),
                    'message': 'photo uploaded',
                    'status': 'success'
                    })
            response.headers.add('Access-Control-Allow-Origin', '*')

            return response

        response = jsonify({
                'data': '',
                'message': 'Something when wrong',
                'status': 'error'
                })
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response

api.add_resource(HelloWorld, '/')
api.add_resource(PhotoUpload,'/upload')

if __name__ == '__main__':
    app.run(debug=True)