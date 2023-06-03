import tensorflow as tf
import dask
from dask.distributed import Client
from multiprocessing import Process, Semaphore
from constants import *
from data_utils import data_utils_class
import models
import os

if __name__ == '__main__':
    dask.config.set(temporary_directory=DASK_TMP_DIR)

    client = Client(n_workers=4, threads_per_worker=2, memory_limit="10GB")
    print(client)
    semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS)
    
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
    
    
    # Apply KFold cross-validation to appliances and train final models
    for household in data_utils.get_households(CUSTOM_METADATA):
        sensors = [Process(target=models.learn,
                           args=(f"models/appliances/household={household}/appliance={sensor}/",
                                 models.get_appliance_ec_input, models.get_appliance_ec_output, train_val_windows, full_train_windows,
                                 (household, sensor), semaphore)
                          ) for sensor in data_utils.get_appliances(CUSTOM_METADATA, household)]
        for sensor in sensors:
            sensor.start()
        for sensor in sensors:
            sensor.join()
            
    print("\n-------------------------------", f"|     Done training appliances!    |", "--------------------------------\n")
    
    print("\n-------------------------------", f"|     Start testing appliances...    |", "--------------------------------\n")
    # Make appliance predictions and aggregate them to all levels above while recording predictions and metrics
    appl_agg_layer = models.ApplianceAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA), data_path=SAVE_DIR)
    appl_agg_layer.from_disk()
    appl_agg_layer.evaluate(test_windows=test_windows)
    print("\n-------------------------------", f"|     Done testing appliances!    |", "--------------------------------\n")


    print("\n-------------------------------", f"|     Start Training households...    |", "--------------------------------\n")
    
    # Apply KFold cross-validation to households and train final models
    
    households = [Process(target=models.learn,
                          args=(f"models/households/household={household}/",
                                models.get_household_ec_input, models.get_household_ec_output, train_val_windows, full_train_windows,
                                (household,), semaphore)
                         ) for household in data_utils.get_households(CUSTOM_METADATA)]
    for household in households:
        household.start()
    for household in households:
        household.join()
    
    print("\n-------------------------------", f"|     Done Training households!    |", "--------------------------------\n")

    print("\n-------------------------------", f"|     Start testing households...    |", "--------------------------------\n")
    # Make household predictions and aggregate them to all levels above while recording predictions and metrics
    house_agg_layer = models.HouseholdAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA), data_path=SAVE_DIR)
    house_agg_layer.from_disk()
    house_agg_layer.evaluate(test_windows=test_windows)

    print("\n-------------------------------", f"|     Done testing households!    |", "--------------------------------\n")


    print("\n-------------------------------", f"|     Start training communities...    |", "--------------------------------\n")

    # Apply KFold cross-validation to communities and train final models
    communities = [Process(target=models.learn,
                          args=(f"models/communities/community={community}/",
                                models.get_community_ec_input, models.get_community_ec_output, train_val_windows, full_train_windows,
                                (community,), semaphore)
                         ) for community in data_utils.get_communities(CUSTOM_METADATA)]
    for community in communities:
        community.start()
    for community in communities:
        community.join()
        
    print("\n-------------------------------", f"|     Done training communities!    |", "--------------------------------\n")

    print("\n-------------------------------", f"|     Start testing communities...    |", "--------------------------------\n")

    # Make community predictions and aggregate them to all levels above while recording predictions and metrics
    comm_agg_layer = models.CommunityAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA), data_path=SAVE_DIR)
    comm_agg_layer.from_disk()
    comm_agg_layer.evaluate(test_windows=test_windows)
    
    print("\n-------------------------------", f"|     Done testing communities!    |", "--------------------------------\n")
    
    
    print("\n-------------------------------", f"|     Start training cities...    |", "--------------------------------\n")
    
    # Apply KFold cross-validation to cities and train final models
    cities = [Process(target=models.learn,
                           args=(f"models/cities/city={city}/",
                                 models.get_city_ec_input, models.get_city_ec_output, train_val_windows, full_train_windows,
                                 (city,), semaphore)
                          ) for city in data_utils.get_cities(CUSTOM_METADATA)]
    for city in cities:
        city.start()
    for city in cities:
        city.join()
    
    print("\n-------------------------------", f"|     Done training cities!    |", "--------------------------------\n")

    print("\n-------------------------------", f"|     Start testing cities...    |", "--------------------------------\n")

    # Make city predictions and aggregate them to all levels above while recording predictions and metrics
    city_agg_layer = models.CityAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA), data_path=SAVE_DIR)
    city_agg_layer.from_disk()
    city_agg_layer.evaluate(test_windows=test_windows)
    
    print("\n-------------------------------", f"|     Done testing cities!    |", "--------------------------------\n")
                
        
        

