import sys
sys.path.append('E:\碳排放时间预测\code\LSTM')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time
import math
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os
from sklearn.ensemble import RandomForestRegressor
import joblib
from config import Config
from EarlyStopping import EarlyStopping
from torch.optim.lr_scheduler import StepLR


import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader


from  model1 import BiLSTM,LSTM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_next(path, sample, args,SEQ_LEN,m,n,epoch=35):
    
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print('predicting...')
    
    temp1 = sample.detach().numpy().tolist()
    for i in range(epoch):
        sample = sample.reshape(1, SEQ_LEN, 1)
        # print(sample)
        pred = model(sample)
        # print(pred.detach().numpy().reshape(1,).tolist())
        # pred = pred.detach().numpy()
        temp1.append(pred.detach().numpy().reshape(1,).tolist())
        sample = torch.tensor(np.array(temp1[i+1 : i+SEQ_LEN+1]),dtype=torch.float32)
    return [(m - n) * p[0] + n for p in  temp1]