import torch
import numpy as np
from model import CNN

SAVE_PATH = 'dt_model.pth'
Generator_PATH = 'generator.pth'
WaterMark_PATH = 'watermark.pth'
threshold = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:\n', device)
cnn = CNN().to(device)
cnn.load_state_dict(torch.load(SAVE_PATH))

X = torch.load(Generator_PATH)
b = torch.load(WaterMark_PATH)

w = cnn.get_parameter('conv2.0.weight')
w = torch.mean(w, dim= 0).view(1, -1)

wm = torch.sigmoid(torch.matmul(w, X))
one = torch.ones_like(wm)
wm = torch.where(wm > threshold, one, wm)
print(torch.where(wm != b)[1])
print('wm: ', wm)
print('b: ' , b)