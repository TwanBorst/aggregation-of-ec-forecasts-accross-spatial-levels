import dask
from dask.distributed import Client
from constants import *
from data_utils import data_utils_class
import models

from result_utils import result_utils_class

if __name__ == '__main__':
    dask.config.set(temporary_directory=DASK_TMP_DIR)

    client = Client(n_workers=4, threads_per_worker=2, memory_limit="10GB")
    print(client)
    
    # Initialise Data Utils Singleton
    data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX)
 
 
    # Generate metadata for available data
    data_utils.generate_metadata(data_path=DATA_GLOB, metadata_files=ORIGINAL_METADATA, save_path=CUSTOM_METADATA)
    
    # Process all the data (includes cleaning) and store it on disk
    data_utils.process_data(files=DATA_GLOB, metadata_files=CUSTOM_METADATA, save_path=SAVE_DIR, from_time=FROM_TIME, end_time=END_TIME)
    
    # Aggregate the data to a household, community and city level and store it on disk
    data_utils.generate_aggregated_data(data_path=SAVE_DIR, metadata=CUSTOM_METADATA)
    
    # Normalize data using Z-score normalization
    data_utils.normalize_data(data_path=SAVE_DIR)
    
    
    print("\n-------------------------------", f"|     Splitting the data...    |", "--------------------------------\n")

    # Split data
    train_val_windows, test_windows = models.split_data(TEST_FRACTION, FOLDS)
    full_train_windows = models.split_data(TEST_FRACTION, 1)[0][0][0]
    
    print("\n-------------------------------", f"|     Done splitting the data!    |", "--------------------------------\n")
    
    print("\n-------------------------------", f"|     Start Training appliances...    |", "--------------------------------\n")
    
    
    models.train_appliances(train_val_windows, full_train_windows)
    models.test_appliances(test_windows)

    models.train_households(train_val_windows, full_train_windows)
    models.test_households(test_windows)

    models.train_communities(train_val_windows, full_train_windows)
    models.test_communities(test_windows)
    
    models.train_cities(train_val_windows, full_train_windows)
    models.test_cities(test_windows)


    result_utils = result_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX)
    result_utils.average_metrics_per_epoch()
    result_utils.summarize_evaluations()
    