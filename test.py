import scipy.io as scio
import csv
import numpy as np

path = 'data/predict.mat'
data = scio.loadmat(path)
result = (data['predict'])
print(result.shape)

f = open('predict_result.csv', 'w', encoding='utf-8',newline='')
csv_writer = csv.writer(f)
for i in range(result.shape[0]):
    print(result[i][0])
    csv_writer.writerow((result[i][0]))


f.close()
