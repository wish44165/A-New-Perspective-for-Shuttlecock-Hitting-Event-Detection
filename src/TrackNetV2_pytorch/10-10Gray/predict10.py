import os
import sys
import json
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from TrackNet10 import TrackNet10
import dataloader10
import numpy as np
from dataloader10 import TrackNetLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2
import math
from PIL import Image
import time
BATCH_SIZE=1
HEIGHT=288
WIDTH=512

parser = argparse.ArgumentParser(description = 'Pytorch TrackNet6')
parser.add_argument('--video_name', type = str, help = 'input video name for predict')
parser.add_argument('--lr', type = float, default = 1e-1, help = 'learning rate (default: 0.1)')
parser.add_argument('--load_weight', type = str, help = 'input model weight for predict')
parser.add_argument('--optimizer', type = str, default = 'Ada', help = 'Ada or SGD (default: Ada)')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default = 1, help = 'random seed (default: 1)')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ',torch.cuda.is_available())

def WBCE(y_pred, y_true):
	eps = 1e-7
	loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) + torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
	return torch.mean(loss)

def custom_time(time):
	remain = int(time / 1000)
	ms = (time / 1000) - remain
	s = remain % 60
	s += ms
	remain = int(remain / 60)
	m = remain % 60
	remain = int(remain / 60)
	h = remain
	#Generate custom time string
	cts = ''
	if len(str(h)) >= 2:
		cts += str(h)
	else:
		for i in range(2 - len(str(h))):
			cts += '0'
		cts += str(h)
	
	cts += ':'

	if len(str(m)) >= 2:
		cts += str(m)
	else:
		for i in range(2 - len(str(m))):
			cts += '0'
		cts += str(m)

	cts += ':'

	if len(str(int(s))) == 1:
		cts += '0'
	cts += str(s)

	return cts

################# video #################

cap = cv2.VideoCapture(args.video_name)
try:
	total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
except:
	total_frames = -1
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


ratio_h = height / HEIGHT
ratio_w = width / WIDTH
size = (width, height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = args.video_name[:-4]+'_predict.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

#########################################

f = open(args.video_name[:-4]+'_predict.csv', 'w')
f.write('Frame,Visibility,X,Y,Time\n')

############### TrackNet ################
model = TrackNet10()
model.to(device)
if args.optimizer == 'Ada':
	optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
	#optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
else:
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
checkpoint = torch.load(args.load_weight)
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']
model.eval()
count = 0
count2 = -10
time_list=[]
start1 = time.time()
while True:
	rets = []
	images = []
	frame_times = []
	for idx in range(10):
		# Read frame from wabcam
		ret, frame = cap.read()
		t = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
		rets.append(ret)
		images.append(frame)
		frame_times.append(t)
		count += 1
		count2 += 1

	if all(rets):
		grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
	elif count >= count:
		break
	else:
		print("read frame error. skip...")
		continue

	# TackNet prediction
	unit = np.stack(grays, axis=2)
	unit = cv2.resize(unit, (WIDTH, HEIGHT))
	unit = np.moveaxis(unit, -1, 0).astype('float32')/255
	unit = torch.from_numpy(np.asarray([unit])).to(device)
	with torch.no_grad():
		start = time.time()
		h_pred = model(unit)
		end = time.time()
		time_list.append(end - start)
	h_pred = h_pred > 0.5
	h_pred = h_pred.cpu().numpy()
	h_pred = h_pred.astype('uint8')
	h_pred = h_pred[0]*255

	for idx_f, (image, frame_time) in enumerate(zip(images, frame_times)):
		show = np.copy(image)
		show = cv2.resize(show, (frame.shape[1], frame.shape[0]))
		# Ball tracking
		if np.amax(h_pred[idx_f]) <= 0: # no ball
			f.write(str(count2 + (idx_f))+',0,0,0,'+frame_time+'\n')
			out.write(image)
		else:
			(cnts, _) = cv2.findContours(h_pred[idx_f].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			rects = [cv2.boundingRect(ctr) for ctr in cnts]
			max_area_idx = 0
			max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
			for i in range(len(rects)):
				area = rects[i][2] * rects[i][3]
				if area > max_area:
					max_area_idx = i
					max_area = area
			target = rects[max_area_idx]
			(cx_pred, cy_pred) = (int(ratio_w*(target[0] + target[2] / 2)), int(ratio_h*(target[1] + target[3] / 2)))
			f.write(str(count2 + (idx_f))+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time+'\n')
			cv2.circle(image, (cx_pred, cy_pred), 5, (0,0,255), -1)
			out.write(image)

f.close()
cap.release()
out.release()
end1 = time.time()
print('Prediction time:', (end1-start1), 'secs')
print('FPS', total_frames / (end1-start1) )
print('Done......')