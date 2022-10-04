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

from model1 import BiLSTM,LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_val_loss(args, model, Val):
    model.eval()
    loss_function1 = nn.MSELoss().to(device)
    loss_sum = 0
    for (seq, label) in Val:
        seq = seq.to(device)
        label = label.to(device)
        y_pred = model(seq)
        loss = loss_function1(y_pred, label)
        loss_sum = loss_sum+loss
    return loss_sum


def train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    early_stopping = EarlyStopping(patience=3)
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    loss_function = nn.MSELoss().to(device)
    print(args.optimizer,args.weight_decay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay
                                     )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = 0
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            # train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                train_loss+=loss

        # validation
        with torch.no_grad():
            val_loss = get_val_loss(args, model, Val)
            # if epoch > min_epochs and val_loss < min_val_loss:
            #     min_val_loss = val_loss
            #     best_model = copy.deepcopy(model)
            early_stopping(val_loss,model,path)
            if early_stopping.early_stop:
                    print("Early stopping")
                    break
            # if train_loss < 0.01:
            #     break

            print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, train_loss, val_loss))
            # print('epoch {:03d} train_loss {:.8f} '.format(epoch, np.mean(train_loss)))
        model.train()

    # state = {'models': best_model.state_dict()}
    # torch.save(state, path)
    

from itertools import chain
def test(args, Dte, path, m, n):
    pred = []
    y = []
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
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    
    
    # print('mape:', get_mape(y, pred))
    # plot
    # x = [i for i in range(1, 151)]
    # x_smooth = np.linspace(np.min(x), np.max(x), 900)
    # y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    # plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    # y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(y, c='black', marker='o', ms=1, alpha=0.75, label='true')
    plt.plot(pred, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    
    return pred