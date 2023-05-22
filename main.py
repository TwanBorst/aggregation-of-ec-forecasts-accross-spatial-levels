import pandas as pd
from dask.distributed import Client
from data_utils import get_data_ids_from_metadata, process_data, generate_metadata

if __name__ == '__main__':
    original_metadata='../data/metadata.csv'
    generated_metadata='../data/generated/metadata.csv'
    data_glob='../data/1s_data*/1s_data*.csv'
    save_dir='../data/generated/'
    save_metadata_path='../data/processed/metadata.csv'
    
    from_timestamp = pd.Timestamp('2019-05-01 05:00:00+00:00', tz="UTC")
    end_timestamp = pd.Timestamp('2019-11-01 04:59:59+00:00', tz="UTC")

    client = Client(n_workers=4, threads_per_worker=2, memory_limit="6GB")
    print(client)
 
    # Uncomment for generating metadata for available data
    # generate_metadata(data_path=generated_metadata, metadata_files=original_metadata, save_path=save_metadata_path)
    
    # Uncomment for processing all the data (includes cleaning) and storing it on disk
    process_data(files=data_glob, metadata_files=generated_metadata, save_path=save_dir, from_time=from_timestamp, end_time=end_timestamp, dumpsite=save_dir)
    


