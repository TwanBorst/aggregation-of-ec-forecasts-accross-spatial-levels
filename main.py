import pandas as pd
from dask.distributed import Client
from data_utils import get_data_ids_from_metadata, process_data, generate_metadata

if __name__ == '__main__':
    metadata_glob='../data/metadata.csv'
    data_glob='../data/1s_data*/1s_data*.csv'
    save_glob='../data/processed/metadata_austin_file1_*.csv'
    save_metadata_path='../data/processed/metadata.csv'
    
    # data_glob='../data/test.csv'

    from_timestamp = pd.Timestamp('2018-06-01 00:00:00-00:00')
    end_timestamp = pd.Timestamp('2019-07-01 00:00:00-00:00')

    client = Client(n_workers=6, threads_per_worker=2, memory_limit="6GB")
    print(client)

    # data_ids = get_data_ids_from_metadata(data_path=metadata_glob, min_1s_completeness=100, min_1s_time=from_timestamp, max_1s_time=end_timestamp)
    # process_data(files=data_glob, save_path=save_glob, sensors=[], data_ids=data_ids, from_time=from_timestamp, end_time=end_timestamp)
    generate_metadata(data_path=data_glob, metadata_files=metadata_glob, save_path=save_metadata_path)


