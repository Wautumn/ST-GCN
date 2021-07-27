# 流量特征处理：补充缺失值、时间周期特征结构

import numpy as np
import pandas as pd

from preprocess.STMatrix import STMatrix
from utils.minmax_normalization import MinMaxNormalization
from utils.utils import *

known_road = get_know_road()
num_known = len(known_road)
time = 5

len_closeness = 5
len_period = 3
len_trend = 1
len_test = 100
external_data = True


def missing_data():
    row_data = pd.read_csv(r'data//features_%smin.csv' % time, header=None)
    data = np.mat(row_data, dtype=np.float32)
    print(data.shape)
    return data


def get_timestamps():
    alltimestamp = []
    for i in range(1, 21):
        time_interval = int(720 / int(time))
        for j in range(time_interval):
            timestamps = "201909" + "%02d" % (i) + str("%04d" % j)
            timestamps = np.array(timestamps)
            alltimestamp.append(timestamps)
    return alltimestamp


def Matrix(len_closeness, len_period, len_trend, len_test=None, external_data=True):
    alldata = missing_data()
    alltimestamp = get_timestamps()
    alltimestamp = np.array(alltimestamp)
    data_all = [alldata]
    timestamps_all = [alltimestamp]
    data_train = alldata[:-len_test]
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        st = STMatrix(data, timestamps)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness, len_period=len_period,
                                                             len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    Y = Y.reshape(-1, 100)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    # XC shape:  (1794, 7, 100) XP shape:  (1794, 3, 100) XT shape:  (1794, 1, 100) Y shape: (1794, 1, 100)

    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape, 'test shape: ', XC_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test


# Matrix(len_closeness, len_period, len_trend, 100, False)

