import pickle
import pandas as pd
import numpy as np
import pmdarima as pm
import torch
from pmdarima.arima import ARIMA
import util
import matplotlib.pyplot as plt
from pmdarima.utils import diff
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs
from pmdarima.preprocessing import LogEndogTransformer
from pmdarima.preprocessing import BoxCoxEndogTransformer
from pmdarima.pipeline import Pipeline
from pmdarima.utils import tsdisplay
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import normaltest
import time


def main():
    als = [100, 30]

    for al in als:
        data_path = f'data/2070hh_al{al}.h5'
        save_path = f'garage/al{al}/exp3_ARIMA/'
        df = pd.read_hdf(data_path)
        fitted_models = []
        unfit_models = []

        # gebruikt hele dataset

        num_samples = df.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.7)
        num_val = num_samples - num_test - num_train

        train = df[:num_train]
        val = df[num_train:num_train+num_val]
        test = df[-num_test:]

        # tsdisplay(train['0'], lag_max=100)
        #
        # y_train_log, _ = LogEndogTransformer(lmbda=1e-6).fit_transform(train['0'])
        # tsdisplay(y_train_log, lag_max=100)
        # print(f'Al: {al} Log: ' + str(normaltest(y_train_log)[1]))
        #
        # y_train_bc, _ = BoxCoxEndogTransformer(lmbda2=1e-6).fit_transform(train['0'])
        # tsdisplay(y_train_bc, lag_max=100)
        # print(f'Al: {al} BC: ' + str(normaltest(y_train_bc)[1]))

        # plt.plot(train['0'].values)
        # plt.plot(val['0'].values)
        # plt.plot(preds_val)
        # plt.show()

        # train op 1 maand, predict 1 week

        # start_date = df.index.get_loc(pd.to_datetime('2013-11-24'))
        # end_date = df.index.get_loc(pd.to_datetime('2013-12-25'))

        # train = df[start_date:end_date]
        # test = df[end_date:]

        # plt.plot(train['0'])
        # plt.show()
        #
        # plot_acf(train['0'], lags=96)
        # plt.show()
        #
        # plot_pacf(train['0'], lags=96)
        # plt.show()

        fitted_models = []
        unfit_models = []
        train_time = []
        val_time = []

        val_mae = []
        val_mape = []
        val_rmse = []

        # TRAINING (AND VALIDATING)

        print(f'Total houses: {len(df.columns.values)}')
        for house_id in df.columns.values:
            print(f'House id: {house_id}')
            try:
                # Train

                # fit = pm.auto_arima(diff_train, m=0,
                #                     seasonal=False, trace=True, error_action='ignore', suppress_warnings=True,
                #                     stepwise=False, random=True, n_fits=50)
                # fit = ARIMA(order=(2, 1, 1), suppress_warnings=True)
                # fit.fit(diff_train)

                # plt.plot(train[house_id])
                # plt.plot(val[house_id])
                # plt.show()

                t1 = time.time()
                fit = Pipeline([
                    ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),
                    ('arima', pm.AutoARIMA(suppress_warnings=True, stepwise=True,
                                           # trace=True, max_p=10, max_q=10
                                           ))
                ])

                fit.fit(train[house_id])
                t2 = time.time()
                train_time.append(t2-t1)
                print('Total training time: {:.4f} secs'.format(t2-t1))
                t1 = time.time()
                with open(f'{save_path}arima_{house_id}.pkl', 'wb') as pkl:
                    pickle.dump(fit, pkl)

                print(fit.get_params())

                fitted_models.append(house_id)

                # Validate
                preds_val = fit.predict(n_periods=val[house_id].shape[0])
                metrics = util.metric(torch.tensor(preds_val), torch.tensor(val[house_id].values))
                log = 'Ag level: {:d} For model: {:s}, val MAE {:.4f}, val MAPE: {:.4f}, val RMSE: {:.4f}'
                print(log.format(al, house_id, metrics[0], metrics[1], metrics[2]))
                val_mae.append(metrics[0])
                val_mape.append(metrics[1])
                val_rmse.append(metrics[2])

                fit.update(val[house_id])
                t2 = time.time()
                val_time.append(t2-t1)
                print("Total inference time: {:.4f} secs".format(t2-t1))

                # plt.plot(train[house_id].values)
                # plt.plot(val[house_id].values)
                # plt.plot(preds_val)
                # plt.show()

            except ValueError:
                unfit_models.append(house_id)

        test_mae = []
        test_mape = []
        test_rmse = []

        # TESTING

        for model in fitted_models:
            with open(f'{save_path}arima_{model}.pkl', 'rb') as pkl:
                # Test

                arima = pickle.load(pkl)
                # real = val[model]
                # diff_val = diff(real_val, lag=24, differences=1)
                # diff_val = diff_val[np.logical_not(np.isnan(diff_val))]

                preds_test = arima.predict(n_periods=test[model].shape[0])

                # Now add the actual samples to the model and create NEW forecasts
                # for _, day_data in test[model].resample('D'):
                #     if day_data.shape[0] == 24:
                #         arima.update(day_data)
                #         new_preds = arima.predict(n_periods=24)
                #         if preds is None:
                #             preds = pd.Series(new_preds, index=day_data.index)
                #         else:
                #             preds = pd.concat([preds, pd.Series(new_preds, index=day_data.index)])
                #         if real is None:
                #             real = day_data
                #         else:
                #             real = pd.concat([real, day_data])

                # preds_val_t = torch.tensor(preds_val)
                # real_test = torch.tensor(real_val

                metrics = util.metric(torch.tensor(preds_test), torch.tensor(test[model].values))
                log = 'Ag level: {:d} For model: {:s}, MAE {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
                print(log.format(al, model, metrics[0], metrics[1], metrics[2]))
                test_mae.append(metrics[0])
                test_mape.append(metrics[1])
                test_rmse.append(metrics[2])
                # fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                # ax.plot(real_val)
                # ax.plot(preds_val)
                # cf = pd.DataFrame(conf_int)
                # ax.fill_between(range(0, len(preds_val)), cf[0], cf[1], alpha=.3)
                # plt.show()
                # plot_arima(real_val, preds_val)

                # in_sample_preds = arima.predict_in_sample()
                # plot_arima(train[model], in_sample_preds)

        log = 'Al: {:d}, Average for all houses: test MAE {:.4f}, test MAPE: {:.4f}, test RMSE: {:.4f}, valid MAE {:.4f}, valid MAPE {:.4f}, valid RMSE {:.4f}'
        print(log.format(al, np.mean(test_mae), np.mean(test_mape), np.mean(test_rmse), np.mean(val_mae), np.mean(val_mape), np.mean(val_rmse)))

        print("Average training time: {:.4f} secs/house".format(np.mean(train_time)))
        print("Average inference time: {:.4f} secs/house".format(np.mean(val_time)))

        res = {'train_time': train_time, 'val_time': val_time,
                'test_mae': test_mae, 'test_mape': test_mape, 'test_rmse': test_rmse,
                'val_mae': val_mae, 'val_mape': val_mape, 'val_rmse': val_rmse}

        with open(f'garage/al{al}/arima_res.pkl', 'wb') as pkl:
            pickle.dump(res, pkl)


def plot_arima(truth, forecasts, title="ARIMA", xaxis_label='Time',
               yaxis_label='Value', c1='#A6CEE3', c2='#B2DF8A'):
    # set up the plot
    plt.title(title)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)

    # add the lines
    # plt.plot(truth.index, truth.values, color=c1, label='Observed')
    # plt.plot(forecasts.index, forecasts.values, color=c2, label='Forecasted')
    plt.plot(truth, color=c1, label='Observed')
    plt.plot(forecasts, color=c2, label='Forecasted')
    plt.legend()

    plt.show()


# in_sample_pred = fit.predict_in_sample()
# in_sample_predT = torch.tensor(in_sample_pred.values)
# house_janT = torch.tensor(house_jan2hh.values)
# print(f'type in sample pred {type(in_sample_pred)}, type house_jan {type(house_jan)}')
# mae, mape, rmse = util.metric(in_sample_predT, house_janT)
# print(f'MAE: {mae}, MAPE: {mape}, RMSE: {rmse}')
#
# next_25 = fit.predict(n_periods=25, X=house_jan2hh)
# actual_next = df['0'][:(32 * 24)]
# plot_arima(actual_next, next_25)

if __name__ == "__main__":
    main()
