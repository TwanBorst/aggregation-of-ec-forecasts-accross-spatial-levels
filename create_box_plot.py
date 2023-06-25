import pickle
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np

# DATA:
# all metrics
def main():
    als = [10, 30, 100, 300, 2070]

    res = {}

    for al in als:
        print(f'Al: {al}')
        # Non Ada GWN

        with open(f'garage/al{al}/nonAda_GNN_res.pkl', 'rb') as pkl:
            df = pickle.load(pkl)

            res[f'nonAda_{al}_mae'] = df['amae']
            res[f'nonAda_{al}_mape'] = df['amape']
            res[f'nonAda_{al}_rmse'] = df['armse']

        # Ada GWN

        print('ADA')

        with open(f'garage/al{al}/AdaGNN_res.pkl', 'rb') as pkl:
            df = pickle.load(pkl)

            res[f'Ada_{al}_mae'] = df['amae']
            res[f'Ada_{al}_mape'] = df['amape']
            res[f'Ada_{al}_rmse'] = df['armse']

        # ARIMA
        print('ARIMA')

        if al == 10:
            df = pd.read_csv(f'garage/al{al}/arima_res.csv')
            res[f'ARIMA_{al}_mae'] = df['MAE'].values.tolist()
            res[f'ARIMA_{al}_mape'] = df['MAPE'].values.tolist()
            res[f'ARIMA_{al}_rmse'] = df['RMSE'].values.tolist()
        else:
            with open(f'garage/al{al}/arima_res.pkl', 'rb') as pkl:
                df = pickle.load(pkl)

                res[f'ARIMA_{al}_mae'] = df['test_mae']
                res[f'ARIMA_{al}_mape'] = df['test_mape']
                res[f'ARIMA_{al}_rmse'] = df['test_rmse']

    for al in als:
        old_count = len(res[f'ARIMA_{al}_rmse'])
        res[f'ARIMA_{al}_rmse'] = [x for x in res[f'ARIMA_{al}_rmse'] if x <= 2]
        diff = old_count - len(res[f'ARIMA_{al}_rmse'])
        for i in range(diff):
            res[f'ARIMA_{al}_rmse'].append(np.mean(res[f'ARIMA_{al}_rmse']))

    # for al in als:
    #     old_count = len(res[f'ARIMA_{al}_mae'])
    #     res[f'ARIMA_{al}_mae'] = [x for x in res[f'ARIMA_{al}_mae'] if x <= 2]
    #     diff = old_count - len(res[f'ARIMA_{al}_mae'])
    #     for i in range(diff):
    #         res[f'ARIMA_{al}_mae'].append(np.mean(res[f'ARIMA_{al}_mae']))

    if al == 2070:
        res['ARIMA_2070_mae'] = [0.1586]
        res['ARIMA_2070_mape'] = [0.3194]
        res['ARIMA_2070_rmse'] = [192.9]

        res['Ada_2070_mae'] = [0.0389]
        res['Ada_2070_mape'] = [0.0789]
        res['Ada_2070_rmse'] = [52.5]

        res['nonAda_2070_mae'] = [0.0325]
        res['nonAda_2070_mape'] = [0.066]
        res['nonAda_2070_rmse'] = [44.1]

    models = ['Ada', 'ARIMA', 'nonAda']
    for al in als:
        if al == 2070:
            continue
        for model in models:
            res[f'{model}_{al}_rmse'] = [i * 1000 for i in res[f'{model}_{al}_rmse']]


    arima = [res['ARIMA_10_rmse'], res['ARIMA_30_rmse'], res['ARIMA_100_rmse'], res['ARIMA_300_rmse'], res['ARIMA_2070_rmse']]
    non_ada = [res['nonAda_10_rmse'], res['nonAda_30_rmse'], res['nonAda_100_rmse'], res['nonAda_300_rmse'], res['nonAda_2070_rmse']]
    ada = [res['Ada_10_rmse'], res['Ada_30_rmse'], res['Ada_100_rmse'], res['Ada_300_rmse'], res['Ada_2070_rmse']]

    # arima = [res['ARIMA_10_mape'], res['ARIMA_30_mape'], res['ARIMA_100_mape'], res['ARIMA_300_mape'], res['ARIMA_2070_mape']]
    # non_ada = [res['nonAda_10_mape'], res['nonAda_30_mape'], res['nonAda_100_mape'], res['nonAda_300_mape'], res['nonAda_2070_mape']]
    # ada = [res['Ada_10_mape'], res['Ada_30_mape'], res['Ada_100_mape'], res['Ada_300_mape'], res['Ada_2070_mape']]

    # arima = [res['ARIMA_10_mae'], res['ARIMA_30_mae'], res['ARIMA_100_mae'], res['ARIMA_300_mae'], res['ARIMA_2070_mae']]
    # non_ada = [res['nonAda_10_mae'], res['nonAda_30_mae'], res['nonAda_100_mae'], res['nonAda_300_mae'], res['nonAda_2070_mae']]
    # ada = [res['Ada_10_mae'], res['Ada_30_mae'], res['Ada_100_mae'], res['Ada_300_mae'], res['Ada_2070_mae']]

    labels_list = ['G10', 'G30', 'G100', 'G300', 'G2070']

    colors = ['#f7fcb9', '#addd8e', '#31a354']

    data_groups = [arima, non_ada, ada]

    width = 1 / len(labels_list)
    xlocations = [x * ((1 + len(data_groups)) * width) for x in range(len(arima))]

    symbol = '+'
    ymin = min([val for dg in data_groups for data in dg for val in data])
    ymax = max([val for dg in data_groups for data in dg for val in data])

    ax = pl.gca()
    ax.set_ylim(ymin, ymax)

    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)

    pl.xlabel('Aggregation level')
    pl.ylabel('RMSE (W)')
    pl.title('Test Data RMSE')

    space = len(data_groups) / 2
    offset = len(data_groups) / 2

    # --- Offset the positions per group:

    group_positions = []
    for num, dg in enumerate(data_groups):
        _off = (0 - space + (0.5 + num))
        print(_off)
        group_positions.append([x + _off * (width + 0.01) for x in xlocations])

    for dg, pos, c in zip(data_groups, group_positions, colors):
        boxes = ax.boxplot(dg,
                           sym=symbol,
                           labels=[''] * len(labels_list),
                           #            labels=labels_list,
                           positions=pos,
                           widths=width,
                           boxprops=dict(facecolor=c),
                           #             capprops=dict(color=c),
                           #            whiskerprops=dict(color=c),
                           #            flierprops=dict(color=c, markeredgecolor=c),
                           medianprops=dict(color='grey'),
                           #           notch=False,
                           #           vert=True,
                           #           whis=1.5,
                           #           bootstrap=None,
                           #           usermedians=None,
                           #           conf_intervals=None,
                           patch_artist=True,
                           )
    ax.set_xticks(xlocations)
    ax.set_xticklabels(labels_list, rotation=0)

    pl.plot([], c='#f7fcb9', label='ARIMA')
    pl.plot([], c='#addd8e', label='nonAda-GWN')
    pl.plot([], c='#31a354', label='Ada-GWN')
    pl.legend()

    pl.savefig('fig/rmse_boxplot.png')
    pl.show()


if __name__ == "__main__":
    main()
