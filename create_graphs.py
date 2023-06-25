import pandas as pd
import matplotlib.pyplot as plt


def main():
    # als = [10, 30, 300, 2070]
    als = [100]
    exp_ids = ['exp1_nonAdaGNN', 'exp2_adaGNN', 'exp3_ARIMA']

    timestamps = list(range(1, 49))
    # timestamps = list(range(1, 193))

    font = {'family': 'Arial', 'size': 20}
    plt.rc('font', **font)
    j = 0
    for al in als:
        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        i = 0
        for exp_id in exp_ids:
            directory = f'garage/al{al}/{exp_id}/wave_al{al}.csv'
            df = pd.read_csv(directory)
            df2 = pd.read_csv(f'garage/al{al}/exp3_ARIMA/wave3_al{al}.csv')
            df['real12'] = [i * 1000 for i in df['real12']]
            df['pred12'] = [i * 1000 for i in df['pred12']]
            df2['real12'] = [i * 1000 for i in df2['real12']]
            df2['pred12'] = [i * 1000 for i in df2['pred12']]

            if i == 0:
                plt.plot(timestamps, df['real12'][-193:-145], label='Real')
            if exp_id == 'exp1_nonAdaGNN':
                plt.plot(timestamps, df['pred12'][-193:-145], label='nonAda-GNN')
            elif exp_id == 'exp2_adaGNN':
                plt.plot(timestamps, df['pred12'][-193:-145], label='Ada-GNN')
            elif exp_id == 'exp3_ARIMA':
                plt.plot(timestamps, df2['pred12'][-192:-144], label='ARIMA')
            plt.ylabel('Energy (W/Dwelling)')
            plt.xlabel('Hour')
            plt.title(f'2013/12/25 - 2013/12/26 (G{al} - {2070//al})')
            # if j == 0:
            plt.legend(fontsize=12)
            i += 1
        # plt.xlim(0,193)
        plt.xlim(0, 49)
        # plt.xticks([i*24 for i in range(9)])
        plt.xticks([i*6 for i in range(9)])
        plt.savefig(f'./fig/al{al}')
        plt.show()
        j += 1




if __name__ == "__main__":
    main()
