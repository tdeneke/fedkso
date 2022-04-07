from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import yaml
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from data.read_data import read_data
from models.trainkso import run
from utils.general import get_latest_run
from utils.torch_utils import de_parallel
from copy import deepcopy

import os
import torch

# TODO: further update this to call the KSO train 
def train(ckptf,data,settings):
    print("-- RUNNING TRAINING --", flush=True)

    # KSO comes with its own data loader 
    # (x_train, y_train) = read_data(data, trainset=True)

    results = run(entity='tdeneke', data=data, hyp=settings, \
        weights=ckptf, project='data', name='exp', batch=3, epochs=10, \
        single_cls=True, workers=4)

    print("-- TRAINING COMPLETED --", flush=True)
    return results

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)
    
    from fedn.utils.pytorchhelper import PytorchHelper
    helper = PytorchHelper()
    weights = helper.load_model(sys.argv[1])
    for name, param in weights.items():
        weights[name] = torch.from_numpy(param)

    # get the latest local check point model file
    ckptf = get_latest_run()
    if not ckptf: 
        # ckptf = 'data/yolov5m.pt'
        ckptf = 'data/yolov5s.pt'

    # update the latest local check point model with global weights
    # ckpt = torch.load(ckptf, map_location=torch.device('cpu'))
    ckpt = torch.load(ckptf)
    from models.mnist_model import create_seed_model
    model = create_seed_model()
    model.load_state_dict(weights, strict=False)  # load
    ckpt['model'] = deepcopy(de_parallel(model)).half()
    torch.save(ckpt, ckptf)

    results = train(ckptf, '../data/koster.yaml', 'data/hyp.scratch.yaml')

    # save weight for global averaging     
    if get_latest_run():
        ckptf = get_latest_run()
    ckpt = torch.load(ckptf)
    helper.save_model(ckpt['model'].state_dict(),sys.argv[2])
