import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def add(row, min):
    return row[0] + min

result = None

k=0
for i in range(111):
    df = pd.read_csv("data/archive/halfhourly_dataset/halfhourly_dataset/block_" + str(i) + ".csv", parse_dates=False)
    print(i)
    if k == 2070:
        break
    for lclid in df['LCLid'].unique():
        if k == 2070:
            break
        cond = df['LCLid'] == lclid
        df1 = df[cond]

        df1.loc[:, "energy(kWh/hh)"] = df1['energy(kWh/hh)'].replace("Null", "0.0")
        df1.loc[:, "energy(kWh/hh)"] = df1['energy(kWh/hh)'].replace("", "0.0")
        df1.loc[:, "energy(kWh/hh)"] = df1['energy(kWh/hh)'].fillna("0.0")
        df1.loc[:, "energy(kWh/hh)"] = df1["energy(kWh/hh)"].astype("float64")
        df1.loc[:, "tstp"] = df1['tstp'].str[:-8]

        list = zip(df1['tstp'], df1['energy(kWh/hh)'])
        index, values = zip(*list)
        df1 = pd.DataFrame({
            lclid: values
        }, index=pd.DatetimeIndex(index))

        df1 = df1.resample('60T').sum()

        start_date = pd.to_datetime('2013-01-01')
        end_date = pd.to_datetime('2013-12-31')
        df1 = df1[start_date:end_date]
        if df1.empty:
            continue
        # print((df1[lclid] == 0).mean())
        if ((df1[lclid] == 0).mean()) > 0:
            continue
        df1 = df1 * 1000
        # plt.plot(df1)
        # plt.show()
        k += 1
        if k % 25 == 0:
            print(k)

        if result is None:
            result = df1
        else:
            result = result.join(df1)



result = result.fillna(0.0)
# result = result.apply(zscore)
print(result.shape)

nullfrac = result.isnull().sum() / result.shape[0]
print("Number of fields with zero NAs:", sum(nullfrac == 0.0))
# print(result[0])
print(result.head())
print(result.shape)

result.to_hdf(f"data/{k}hh.h5", key='df', mode='w')