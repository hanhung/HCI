import time
import numpy as np

import werkzeug, os
from flask import Flask
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

            return_bbox = np.load('./example/return_bbox.npy')
            return_label = np.load('./example/return_label.npy')

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
    app.run(debug=True)