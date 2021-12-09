
"""
import tensorflow

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import os


import tempfile
"""
import yaml 
import logging
import torch
from models.yolo import Model

from utils.general import colorstr
from utils.torch_utils import select_device
logger = logging.getLogger(__name__)

# Create an initial Model
def create_seed_model():
	single_cls = True
	hyp = 'data/hyp.scratch.yaml'
	data = '../data/koster.yaml' # dataset
	cfg = ''
	weights = 'data/yolov5m.pt'
	
	# Hyperparameters
	if isinstance(hyp, str):
		with open(hyp) as f:
			hyp = yaml.safe_load(f)  # load hyps dict
	# logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
	
	with open(data) as f:
		data_dict = yaml.safe_load(f)  # data dict
		
	nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
	bs = 16 # opt.batch_size
	dev = '' # opt.device
	device = select_device(dev, batch_size=bs)
	
	ckpt = torch.load(weights, map_location=device)
	model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
	
	return model