import io
import time
import random
import numpy as numpy
from edgetpu.detection.engine import DetectionEngine

from PIL import Image
from PIL import ImageDraw
from io import BytesIO

class ObjectDetection(object):

    def __init__(self, config):
        self.config = config
        with open(self.config['label_path'], 'r', encoding="utf-8") as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)
        self.engine = DetectionEngine(self.config['model_path'])

    async def detect(self, d, cx):
        start_time = time.time()

        # _, width, height, channels = self.engine.get_input_tensor_shape()

        #img = Image.open(si)
        img = Image.open(io.BytesIO(d['frame']))

        ans = self.engine.detect_with_image(img, threshold=self.config['confidence'], keep_aspect_ratio=True, relative_coord=False, top_k=10)
        mats = []
        print(d['id'])
        if ans:
            for v in ans:
                print ('-----------------------------------------')
                label = self.labels[v.label_id]
                if label == 'bus':
                    label = 'car'
                print(label, 'score = ', v.score)
                box = v.bounding_box.flatten().tolist()
                mats.append({
                    'x': float(box[0]),
                    'y': float(box[1]),
                    'width': float(box[2]),
                    'height': float(box[3]),
                    'tag': label,
                    'confidence': float(v.score)
                })
        isObjectDetectionSeparate = d['mon']['detector_pam'] == '1' and d['mon']['detector_use_detect_object'] == '1'
        width = 0
        height = 0
        if isObjectDetectionSeparate and d['mon']['detector_scale_y_object']:
            width = d['mon']['detector_scale_y_object']
        else:
            width = d['mon']['detector_scale_y']
        width = float(width)

        if isObjectDetectionSeparate and d['mon']['detector_scale_x_object']:
            height = d['mon']['detector_scale_x_object']
        else:
            height = d['mon']['detector_scale_x']
        height = float(height)

        resp_time = time.time() - start_time
        print(resp_time)
        await cx({
            'f': 'trigger',
            'id': d['id'],
            'ke': d['ke'],
            'details': {
                'plug': self.config['plug'],
                'name': self.config['plug'],
                'reason': 'object',
                'matrices': mats,
                'imgHeight': width,
                'imgWidth': height,
                'time': resp_time
            }
        })