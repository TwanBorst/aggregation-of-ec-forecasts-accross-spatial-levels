import pandas as pd
from dask.distributed import Client
from data_utils import get_data_ids_from_metadata, process_data

if __name__ == '__main__':
    metadata_glob='../data/1s_data_*/metadata.csv'
    data_glob='../data/1s_data_*/1s_data_*.csv'
    save_glob='../data/processed/1s_data_partition_*.csv'

    from_timestamp = pd.Timestamp('2018-06-01 00:00:00-00:00')
    end_timestamp = pd.Timestamp('2019-07-01 00:00:00-00:00')

    client = Client(n_workers=8, threads_per_worker=2, memory_limit="15GB")
    print(client)

    data_ids = get_data_ids_from_metadata(data_path=metadata_glob, min_1s_completeness=100, min_1s_time=from_timestamp, max_1s_time=end_timestamp)
    process_data(files=data_glob, save_path=save_glob, sensors=[], data_ids=data_ids, from_time=from_timestamp, end_time=end_timestamp)


