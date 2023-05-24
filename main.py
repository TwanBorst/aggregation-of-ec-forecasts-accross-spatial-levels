import pandas as pd
from dask.distributed import Client
from data_utils import data_utils_class

if __name__ == '__main__':
    original_metadata='../data/metadata.csv'
    custom_metadata ='../data/generated/metadata_15min.csv'
    data_glob='../data/15minute_data*/15minute_data*.csv'
    save_dir='../data/generated/15min/'
    
    from_timestamp = pd.Timestamp('2019-05-01 05:00:00+00:00', tz="UTC")
    end_timestamp = pd.Timestamp('2019-11-01 04:45:00+00:00', tz="UTC")

    client = Client(n_workers=4, threads_per_worker=2, memory_limit="8GB")
    print(client)
    
    data_utils = data_utils_class(time_column_name='local_15min', time_frequency='15min', metadata_time_prefix='egauge_1min_')
 
    # Generate metadata for available data
    data_utils.generate_metadata(data_path=data_glob, metadata_files=original_metadata, save_path=custom_metadata)
    
    # Process all the data (includes cleaning) and store it on disk
    data_utils.process_data(files=data_glob, metadata_files=custom_metadata, save_path=save_dir, from_time=from_timestamp, end_time=end_timestamp)
    
    # Aggregate the data to a household, community and city level and store it on disk
    data_utils.generate_aggregated_data(data_path=save_dir, metadata=custom_metadata)


