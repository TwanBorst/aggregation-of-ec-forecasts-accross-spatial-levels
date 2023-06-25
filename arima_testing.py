import pickle
import pmdarima as pm
import pandas as pd
import torch
import util
import matplotlib.pyplot as plt


def main():
    # als = [10, 30, 100, 300, 2070]
    als = [100]
    for al in als:
        print(al)
        data_path = f'data/2070hh_al{al}.h5'
        model_path = f"garage/al{al}/exp3_ARIMA/"
        df = pd.read_hdf(data_path)

        num_samples = df.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.7)
        num_val = num_samples - num_test - num_train

        train = df[:num_train]
        val = df[num_train:num_train + num_val]
        test = df[-num_test:]

        timestamps = list(range(1, 193))

        model = (2070 // al) - 2
        with open(f'{model_path}arima_{model}.pkl', 'rb') as pkl:
            arima = pickle.load(pkl)
            arima.update(val[str(model)])

            # preds = arima.predict(n_periods=test[str(model)].shape[0])
            # real = test[str(model)]

            # Now add the actual samples to the model and create NEW forecasts

            preds = None
            real = None
            for _, day_data in test[str(model)].resample('D'):
                if day_data.shape[0] == 24:
                    arima.update(day_data)
                    new_preds = arima.predict(n_periods=24)
                    if preds is None:
                        preds = pd.Series(new_preds, index=day_data.index)
                    else:
                        preds = pd.concat([preds, pd.Series(new_preds, index=day_data.index)])
                    if real is None:
                        real = day_data
                    else:
                        real = pd.concat([real, day_data])

            # metrics = util.metric(torch.tensor(preds.values), torch.tensor(real.values))

            # metrics = util.metric(torch.tensor(preds), torch.tensor(real))
            # log = 'Ag level: {:d} For model: {:d}, MAE {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
            # print(log.format(al, model, metrics[0], metrics[1], metrics[2]))
            #
            # plt.plot(timestamps, preds[-192:], label='Pred')
            # plt.plot(timestamps, real[-192:], label='Real')
            # plt.show()

            df2 = pd.DataFrame({'real12': real, 'pred12': preds})
            df2.to_csv(model_path + f'wave3_al{al}.csv', index=False)


if __name__ == "__main__":
    main()
