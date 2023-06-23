import dask
import dask.dataframe as dd
import pandas as pd

from constants import *
from dask.distributed import Client

TIME = "15minute"


def main():
    dask.config.set(temporary_directory=DASK_TMP_DIR)

    client = Client(n_workers=4, threads_per_worker=2, memory_limit="10GB")
    print(client)

    print("\n----------------------------", f"|        Getting stats         |", "----------------------------\n")

    metadata = pd.read_csv(SAVE_PATH + f"metadata_{TIME}.csv")

    dataids = metadata['dataid'].values.tolist()
    dataids.sort()

    total = len(dataids)

    for progress, dataid in enumerate(dataids):

        print("\n----------------------------", f"|  Starting with dataid = {dataid}  |",
              "----------------------------")
        print("----------------------------", f"|        progress: {progress}/{total}         |",
              "----------------------------\n")

        # Reads the data related to one dataid / house and sets the index to the time
        dataid_ddf = dd.read_parquet(
            SAVE_PATH + "final_household/",
            filters=[('dataid', '==', dataid)],
            columns=['total']
        )

        dataid_ddf['total'] = dataid_ddf['total'] * 1000

        mean = dataid_ddf.compute().mean(axis=0)                         # The mean value of the house.
        std = dataid_ddf.compute().std(axis=0)                           # The standard deviation of the house.

        print(f"Mean for {dataid} =\t\t", int(round(mean)))
        print(f"Std for {dataid} =\t\t", int(round(std)))

        print("\n----------------------------", f"|    Done with dataid = {dataid}     |",
              "----------------------------\n")



if __name__ == '__main__':
    main()