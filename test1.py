import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import subprocess
from pmdarima.preprocessing import BoxCoxEndogTransformer
from pmdarima.pipeline import Pipeline
from pmdarima.arima.stationarity import  ADFTest
import pmdarima as pm
from pmdarima.utils import diff
import csv
import seaborn as sns



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# df = pd.read_hdf("data/metr-la.h5")
# print(df.head())
# print(df.shape)
# print(df.columns)
#
# plt.plot(df['773869'])
# plt.show()
# #
# df = pd.read_hdf("data/207hh.h5")
# #
# print(df.head())
# print(df.shape)
# print(df.columns)
#
# data = {}
# cat_data = np.load("data/METR-LA/train.npz")
# # cat_data = np.load("data/207HH/train.npz")
# data["x_train"] = cat_data['x']
# data["y_train"] = cat_data['y']
#
# print(type(data))
# print(len(data['x_train']))
# print(cat_data['y_offsets'])
# print(data["x_train"])

# # df = pd.read_csv("wave.csv")
#
# # ax = plt.axes()
# # plt.plot(df['real12'])
# # plt.plot(df['pred12'])
# # plt.plot(df['MAC000002'])
# # plt.plot(df['pred3'])
# # plt.legend(['real12', 'pred12', 'real3', 'pred3'])
# plt.show()
# # plt.savefig("./test" + '.pdf')


# def load_pickle(pickle_file):

# pickle_file = 'data/graph/adj_mat_al100.pkl'

# try:
#     with open(pickle_file, 'rb') as f:
#         pickle_data = pickle.load(f)
# except UnicodeDecodeError as e:
#     with open(pickle_file, 'rb') as f:
#         pickle_data = pickle.load(f, encoding='latin1')
# except Exception as e:
#     print('Unable to load data ', pickle_file, ':', e)
#     raise

# pickle_data = pd.read_pickle(pickle_file)
#
# print(pickle_data)


#     return pickle_data

# agg_level = 3

# df = pd.read_hdf(f"data/2070hh_al{agg_level}.h5")

# correlations = df.corr()
#
# result = pd.DataFrame()
#
# # k=0
# # print(len(df.columns)**2)
# # for col1 in df.columns:
# #     for col2 in df.columns:
# #         if col1 != col2:
# #             # distance, _ = fastdtw(df[col1].values, df[col2].values)
# #             # distance = np.corrcoef(df[col1].values, df[col2].values)[0, 1]
# #             distance = euclidean(df[col1].values, df[col2].values)
# #             new_row = {'from': col1, 'to': col2, 'distance': distance}
# #             result = pd.concat([result, pd.DataFrame(new_row, index=[k])])
# #             k += 1
# #             if k % 250 == 0:
# #                 print(k)
# #
# #     #
# #     # for j in range(i + 1, len(correlations.columns)):
# #     #     new_row = {'from': correlations.columns.values[i], 'to': correlations.columns.values[j], 'distance': correlations.iat[i, j] * 10000}
# #     #     result = pd.concat([result, pd.DataFrame(new_row, index=[k])])
# #     #     k += 1
# #
# # print(result.columns[0])
# #
# # result.to_csv(f"data/graph/distances_al{agg_level}.csv")

# df = pd.read_hdf('data/2070hh_Wh_al300.h5')
#
# ids = ''
#
# for id in df.columns:
#     ids += id
#     ids += ','
#
# print(ids)

# agg_level = 3
#
# df = pd.read_csv(f'data/graph/distances_al{agg_level}.csv')
#
# df = df.replace(',', ';', regex=True)
#
# df = df.drop('Unnamed: 0', axis=1)
#
# df.to_csv(f'data/graph/distances_al{agg_level}.csv')


# als = [10, 30, 100, 300, 2070]
#
# for al in als:
#     print(al)
#     sensor_ids = f'data/graph_W/ids_al{al}_Wh.txt'
#     distances = f'data/graph_W/distances_al{al}.csv'
#     output= f'data/graph_W/adj_mat_al{al}_Wh.pkl'
#     arguments = [f'--sensor_ids_filename={sensor_ids}', f'--distances_filename={distances}', f'--output_pkl_filename={output}']
#     command = ['python', 'gen_adj_mx.py'] + arguments
#     subprocess.run(command)

# --sensor_ids_filename=data/graph/ids_al100.txt
# --distances_filename=data/graph/distances_al100.csv
# --output_pkl_filename=data/graph/adj_mat_al100.pkl

# data_path = 'data/2070hh_al10.h5'
# df = pd.read_hdf(data_path)
#
# num_samples = df.shape[0]
# num_test = round(num_samples * 0.2)
# num_train = round(num_samples * 0.7)
# num_val = num_samples - num_test - num_train
#
# train = df[:num_train]
# val = df[num_train:num_train+num_val]
# test = df[-num_test:]
#
# diff_test = diff(train['0'], lag=24, differences=1)
#
# adf_test = ADFTest(alpha=0.5)
# p_val, should_diff = adf_test.should_diff(diff_test)
# print(f'{p_val}, {should_diff}')
#
# fit = Pipeline([
#     ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),
#     ('arima', pm.AutoARIMA(suppress_warnings=True, stepwise=True,
#                            d=1, trace=True))
# ])
# fit.fit(diff_test)
#

# converting arima results:
# ----------------------------------

# j=0
#
# for i in range(0, 207):
#     save_path = f'garage/al10/exp3_ARIMA/arima_{i}.pkl'
#     with open(save_path, 'rb') as pkl:
#         arima = pickle.load(pkl)
#         # model = getattr(arima, 'named_steps')['arima']
#         # print(model.model_)
#         model = arima.named_steps['arima'].model_
#         print(f'Model: {i}; params: {model}')
#         if str(model) == ' ARIMA(2,1,2)(0,0,0)[0] intercept':
#             print('!!!!!!')
#             j += 1
#
# print(f'total (2,1,2) intercepts: {j}')

# ---------------------------

# als = [30, 100, 300, 2070]
# # als = [10]
# for al in als:
#     with open(f'garage/al{al}/arima_res.pkl', 'rb') as pkl:
#         res = pickle.load(pkl)
#
#         # res = pd.read_csv('garage/al10/arima_res.csv')
#
#         test = list(zip(res['test_mae'], res['test_mape'], res['test_rmse']))
#         # test = list(zip(res['MAE'], res['MAPE'], res['RMSE']))
#         filtered_test = [tup for tup in test if not ((tup[0] + tup[1] + tup[2]) > 5)]
#
#         dropped = len(test) - len(filtered_test)
#         # if al != 30:
#         #     for tup in filtered_test:
#         #         print(f'Sum: {tup[0]+tup[1]+tup[2]}, 0: {tup[0]}, 1: {tup[1]}, 2: {tup[2]}')
#
#         mae = sum(x[0] for x in filtered_test) / len(filtered_test)
#         mape = sum(x[1] for x in filtered_test) / len(filtered_test)
#         rmse = sum(x[2] for x in filtered_test) / len(filtered_test)
#
#         print('Al: {:d}, MAE: {:.4f}, MAPE {:.4f}, RMSE: {:.4f}'.format(al, mae, mape, rmse))
#         print('Num errors: {:d}'.format(dropped))
#
#
#
#
# df = sns.load_dataset('iris')
# df.head()
#
# sns.boxplot( x=df["species"], y=df["sepal_length"] );
# plt.show()
#
# print(df)

als = [10, 30, 100, 300, 2070]

res = {}

for al in als:
    print(f'Al: {al}')

    with open(f'garage/al{al}/nonAda_GNN_res.pkl', 'rb') as pkl:
        df = pickle.load(pkl)
        res[f'nonAda_{al}'] = df['training_time']

    with open(f'garage/al{al}/AdaGNN_res.pkl', 'rb') as pkl:
        df = pickle.load(pkl)
        res[f'Ada_{al}'] = df['training_time']

    print(res)