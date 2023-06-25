import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/2070hh_Wh.h5')
parser.add_argument('--aggregation_level', type=int, default=10)

args = parser.parse_args()


def main():
    print(args)

    data_path = args.data
    agg_level = args.aggregation_level

    df = pd.read_hdf(data_path)

    result = None

    i = 0
    while len(df.columns.values) >= agg_level:
        ids = random.sample(list(df.columns.values), agg_level)
        string = ';'.join(ids)

        columns_sum = df[ids].sum(axis=1)

        df = df.drop(ids, axis=1)
        df1 = columns_sum.to_frame(name=str(i))
        df1 = df1 / agg_level

        if result is None:
            result = df1
        else:
            result = result.join(df1)

        i += 1

    print(result.shape)

    result.to_hdf(f"data/2070hh_Wh_al{agg_level}.h5", key='df', mode='w')

if __name__ == "__main__":
    main()