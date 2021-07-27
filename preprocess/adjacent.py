# 获取100个路口的邻接矩阵，以及10个预测路口的临近路口
import numpy as np
from utils.utils import *

known_road = get_know_road()
num_known = len(known_road)
predict_road = get_predict_road()
num_predict = len(predict_road)


# 邻接矩阵
def get_adjacent_matrix():
    adj = np.zeros([num_known, num_known])
    csvFile = open("data//roadnet.csv", "r", encoding='UTF-8')
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        up = (item[0])
        down = (item[1])
        if up in known_road and down in known_road:
            loa_up = known_road.index(up)
            loa_down = known_road.index(down)
            adj[loa_up, loa_down] = 1
            adj[loa_down, loa_up] = 1
    csvFile.close()
    return adj


# 10个预测路口的相邻路口
def get_predict_adajent():
    adj_predict = dict()
    for road in predict_road:
        adj_predict[road] = set()
    csvFile = open("..//data//roadnet.csv", "r", encoding='UTF-8')
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        up = (item[0])
        down = (item[1])
        if up in known_road and down in predict_road:
            adj_predict[down].add(up)
        if down in known_road and up in predict_road:
            adj_predict[up].add(down)
    csvFile.close()
    print(adj_predict)
    return adj_predict
