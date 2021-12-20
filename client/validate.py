import sys
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from data.read_data import read_data
from utils.general import get_latest_run
from utils.torch_utils import de_parallel
from copy import deepcopy
from models.test import run

import json
from sklearn import metrics
import os
import torch
import yaml
import numpy as np

# TODO: update this to call the KSO validate/test 
def validate(ckptf,data):
    print("-- RUNNING VALIDATION --", flush=True)
     
    try:
        # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        results, m, _ = run(data=data, weights=ckptf, single_cls=True, \
            save_dir='data/expv', project='data', name='expv', conf_thres=0.5)

        print('Training accuracy & loss:', results)

        # model_score_test = model.evaluate(x_test, y_test, verbose=0)
        tresults, tm, _ = run(data=data, weights=ckptf, single_cls=True, \
            save_dir='data/expt', project='data', name='expt', task='test', conf_thres=0.5)


        print('Test accuracy & loss:', tresults)

    except Exception as e:
        print("failed to validate the model {}".format(e),flush=True)
        raise

    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    report = {
                "P": results[0],
                "R": results[1],
                "mAP1": results[2],
                "mAP2": results[3],
                "box": results[4],
                "obj": results[5],
                "cls": results[6],
                "tP": tresults[0],
                "tR": tresults[1],
                "tmAP1": tresults[2],
                "tmAP2": tresults[3],
                "tbox": tresults[4],
                "tobj": tresults[5],
                "tcls": tresults[6],
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    # read and load global weights in the right format
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
    
    # update the latest local check point with global weights
    # ckpt = torch.load(ckptf, map_location=torch.device('cpu'))
    ckpt = torch.load(ckptf)
    from models.mnist_model import create_seed_model
    model = create_seed_model()
    model.load_state_dict(weights, strict=False)  # load
    ckpt['model'] = deepcopy(de_parallel(model)).half()
    torch.save(ckpt, ckptf)
    
    report = validate(ckptf, '../data/koster.yaml')

    with open(sys.argv[2],"w") as fh:
        fh.write(json.dumps(report))