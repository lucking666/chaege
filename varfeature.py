import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import  summary
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import joblib
# from sklearn.externals import joblib
from utils.sgdr import CosineAnnealingLR_with_Restart
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data_path = 'Qclin_first_100_cycle_data_hybird.pt'

data_set = torch.load(data_path)


for name in data_set.keys():
    data_set[name]=np.nan_to_num(data_set[name], nan=0.0)
    data_set[name][np.isinf(data_set[name])] = 0.0

# print(data_set.keys())
# ['train_x', 'train_x_fc', 'train_y', 'eva_x', 'eva_x_fc', 'eva_y', 'test_x_pri', 'test_x_pri_fc', 'test_y_pri', 'test_x_sec', 'test_x_sec_fc', 'test_y_sec']
print("get data is ok")

X_train=data_set['train_x_fc']
y_train=data_set['train_y']
X_test=data_set['test_x_pri_fc']
y_test=data_set['test_y_pri']


# Y_train=np.log10(y_train)
# Y_test=np.log10(y_test)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

# 4. 预测寿命
y_pred = model.predict(X_test)

# 5. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"均方误差 (MSE): {mse}")
print(f"均方根误差 (RMSE): {rmse}")


