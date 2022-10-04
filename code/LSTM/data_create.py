from unicodedata import name
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
# from config import Config
# from EarlyStopping import EarlyStopping
from torch.optim.lr_scheduler import StepLR


import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader


class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, item):
                return self.data[item]

            def __len__(self):
                return len(self.data)


class Data_create():
    def __init__(self,data,train_B,Val_B,Test_B,t,Dtr_p,Val_p1,Val_p2,Dte_p):
        self.data = data
        self.train_B = train_B
        self.Val_B = Val_B
        self.Test_B = Test_B
        self.t = t
        self.Dtr_p = Dtr_p
        self.Val_p1 = Val_p1
        self.Val_p2 = Val_p2
        self.Dte_p = Dte_p
        
        
    
    def nn_seq_us(self,):
        print('data processing...')
        dataset = self.data
        # split
        # train = dataset[:int(len(dataset) * 0.8)]
        # val = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
        # test = dataset[int(len(dataset) * 0.9):len(dataset)]
        m, n = np.max(dataset), np.min(dataset)
        print(m,n)
        def process(data, batch_size, shuffle):
            load1 = data.values
            load = (load1 - n) / (m - n)
            # print(load)
            seq = []
            for i in range(len(data) - self.t):
                train_seq = []
                train_label = []
                for j in range(i, i + self.t):
                    x = [load[j]]
                    train_seq.append(x)
                # for c in range(2, 8):
                #     train_seq.append(data[i + 24][c])
                train_label.append(load[i + self.t])
                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seq.append((train_seq, train_label))

            print(len(seq))
            
            # seq = MyDataset(seq)
            # seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

            return seq

    
        seq = process(dataset,self.train_B,False)
        Dtr = DataLoader(dataset=MyDataset(seq[:int(len(seq) * self.Dtr_p)])
                         , batch_size=self.train_B, shuffle=False, num_workers=0, drop_last=True)
        
        
        Val = DataLoader(dataset=MyDataset(seq[int(len(seq) * self.Val_p1):int(len(seq) * self.Val_p2)])
                         , batch_size=self.Val_B, shuffle=False, num_workers=0, drop_last=True)
        
        Dte = DataLoader(dataset=MyDataset(seq[int(len(seq) * self.Dte_p):len(seq)])
                         , batch_size=self.Test_B, shuffle=False, num_workers=0, drop_last=True)
        return Dtr, Val, Dte, m, n
    
if __name__ == "__main__":
    data = pd.read_excel("E:\碳排放时间预测\data\碳排放总量.xlsx",index_col=0).iloc[:,0]
    Dtr, Val, Dte, m, n = Data_create(data= data
                ,train_B=3
                ,Val_B=3
                ,Test_B=1
                ,t=6
                ,Dtr_p=0.8
                ,Val_p1=0.8
                ,Val_p2=1
                ,Dte_p=0).nn_seq_us()
    print(len(list(Dtr)),len(list(Val)),len(list(Dte)))


