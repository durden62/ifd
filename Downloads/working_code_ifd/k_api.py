import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from PIL import Image
import io

import torch
import torch.backends.cudnn as cudnn
from numpy import random, ascontiguousarray
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from flask import Flask, jsonify, request, send_from_directory
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
def detect(source, save_img=False):
    weights= 'final_weights.pt'
    imgsz = 832
    # Padded resize
    img = letterbox(source, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = ascontiguousarray(img)

    set_logging()
    device = select_device('')

    model = attempt_load(weights, map_location=device)
    check_img_size(imgsz, s = model.stride.max())

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0,255) for _ in range(3)] for _ in range(len(names))]

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

    for i, det in enumerate(pred):

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, source, label=label, color=colors[int(cls)], line_thickness=3)

    # cv2.imshow('abc', source)
    # cv2.waitKey(5000)
    return source


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict1():
        # we will get the file from the request
    file = request.files['file']
    # convert that to bytes
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    results = detect(np.array(img))

    final_img = Image.fromarray(results)
    final_img.save('final.jpg')
    return send_from_directory('.', 'final.jpg', as_attachment=True)

if __name__ == '__main__':
   app.run(debug=True)
