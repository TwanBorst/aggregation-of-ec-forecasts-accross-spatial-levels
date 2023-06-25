import pandas as pd
import os
from scipy.spatial.distance import euclidean
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('--agg_level', type=int, default=3)
#
# args = parser.parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    # print(args)
    #
    # agg_level = args.agg_level

    als = [10, 30, 100, 300, 2070]

    for al in als:

        df = pd.read_hdf(f"data/2070hh_Wh_al{al}.h5")

        result = pd.DataFrame()

        k = 0
        print(len(df.columns)**2)
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    # distance, _ = fastdtw(df[col1].values, df[col2].values)
                    # distance = np.corrcoef(df[col1].values, df[col2].values)[0, 1]
                    # distance = euclidean(df[col1].values, df[col2].values)
                    distance = 0.0
                    new_row = {'from': col1, 'to': col2, 'distance': distance}
                    result = pd.concat([result, pd.DataFrame(new_row, index=[k])])
                    k += 1
                else:
                    distance = 1
                    new_row = {'from': col1, 'to': col2, 'distance': distance}
                    result = pd.concat([result, pd.DataFrame(new_row, index=[k])])
                if k % 2000 == 0:
                    print(k)

        result.to_csv(f"data/graph_W/distances_al{al}.csv")


if __name__ == "__main__":
    main()

# To create the ids:

# ids = ''
#
# for id in df.columns:
#     ids += id
#     ids += ','
#
# print(ids)
