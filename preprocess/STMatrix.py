from __future__ import print_function
from datetime import datetime

import pandas as pd
import numpy as np


# feature data
def string2timestamp(strings, T):
    timestamps = []
    time_per_slot = T  # minute
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot // 60 + 7),
                                                minute=int(
                                                    (slot * time_per_slot - int(slot * time_per_slot // 60) * 60)
                                                )
                                                )
                                       ))
    return timestamps


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, CheckComplete=False, T=5):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = int(T)
        self.pd_timestamps = string2timestamp(timestamps, self.T)
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def add_index(self, ts, newdata):
        self.get_index[ts] = len(self.data)
        self.data = np.row_stack((self.data, newdata))
        self.pd_timestamps = np.append(self.pd_timestamps, ts)

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:

            if d not in self.get_index.keys():
                return False
        return True

    def check_it_one(self, depends):
        if depends[0] not in self.get_index.keys():
            return False
        return True

    def check_one(self, depends):
        if depends not in self.get_index.keys():
            return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version
        """
        print("a")
        offset_frame = pd.DateOffset(minutes=5)
        offset_day = pd.DateOffset(days=1)
        offset_week = pd.DateOffset(days=7)

        XC = []
        XP = []
        XT = []
        Y = []

        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   range(1, len_period + 1),
                   range(1, len_trend + 1)]

        all_interval = int(720 / self.T)
        i = max(all_interval * TrendInterval * len_trend, all_interval * PeriodInterval * len_period, len_closeness)

        while i < len(self.pd_timestamps):
            x_cframe = []
            for j in range(len_closeness):
                x_cframe.append(self.pd_timestamps[i] - j *offset_frame)

            # Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depends[0]])
            Flag = self.check_it(x_cframe)

            # 前段时间是否存在
            if Flag is False:
                i = i + 1
                continue
            else:
                # x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
                x_c = [self.get_matrix(x_cframe[j]) for j in range(len(x_cframe))]

            x_pframe = []
            num_p = 1
            while len(x_pframe) < len_period:
                if (self.check_one(self.pd_timestamps[i] - num_p * offset_day) == True):
                    x_pframe.append(self.pd_timestamps[i] - num_p * offset_day)
                    num_p = num_p + 1
                else:
                    num_p = num_p + 1
            x_p = [self.get_matrix(x_pframe[i]) for i in range(len_period)]

            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_week) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])

            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)

        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y

    def create_test_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version
        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=12 * 60 // self.T)
        offset_day = pd.DateOffset(days=1)
        offset_week = pd.DateOffset(days=7)
        offset_intervalday = pd.DateOffset(hours=12)
        XC = []
        XP = []
        XT = []
        Y = []
        # [all,day,week]
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [1 * 1 * j for j in range(1, len_period + 1)],
                   [1 * 1 * j for j in range(1, len_trend + 1)]]

        print("depends shape is :")
        print(depends[0])
        print(depends[1])
        print(depends[2])
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        print(len(self.pd_timestamps))
        i = 2735
        # print(self.check_it(self.pd_timestamps[2592]-1 * offset_day for j in depends[1]))
        x_c = [self.get_matrix(self.pd_timestamps[2592] - (j) * offset_day) for j in depends[1]]
        x_p = [self.get_matrix(self.pd_timestamps[2592] - (j - 1) * offset_day) for j in depends[1]]
        x_t = [self.get_matrix(self.pd_timestamps[2592] - j * offset_week + offset_day) for j in depends[2]]
        XC.append(np.vstack(x_c))
        XP.append(np.vstack(x_p))
        XT.append(np.vstack(x_t))
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        # Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape)
        return XC, XP, XT


if __name__ == '__main__':
    pass
