from multiprocessing import Process, Semaphore

import dask.dataframe as dd
import numpy as np

import models
from aggregation_layer import (ApplianceAggregationLayer,
                               HouseholdAggregationLayer)
from constants import *
from data_utils import (data_utils_class, get_appliance_ec_input,
                        get_appliance_ec_output, get_city_ec_output,
                        get_community_ec_input, get_community_ec_output,
                        get_household_ec_input, get_household_ec_output)
from result_utils import result_utils_class


def preprocess_data():
    # Initialize data_utils
    data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX)
    
    # Generate metadata for available data
    data_utils.generate_metadata(data_path=DATA_GLOB, metadata_files=ORIGINAL_METADATA, save_path=CUSTOM_METADATA)
    
    # Process all the data (includes cleaning) and store it on disk
    data_utils.process_data(files=DATA_GLOB, metadata_files=CUSTOM_METADATA, save_path=SAVE_DIR, from_time=FROM_TIME, end_time=END_TIME)
    
    # Aggregate the data to a household, community and city level and store it on disk
    data_utils.generate_aggregated_data(data_path=SAVE_DIR, metadata=CUSTOM_METADATA)
    
    # Normalize data using Z-score normalization
    data_utils.normalize_data(data_path=SAVE_DIR)


def epochs_25_without_hyperparameters():
    # Split data
    train_val_windows, test_windows = models.split_data(TEST_FRACTION, 1)
    full_train_windows = train_val_windows[0][0]
    
    epochs = 25
    model_dir_name = "models_25_no_hp"
    
    models.train_appliances(train_val_windows, full_train_windows, epochs=25, model_dir_name=model_dir_name)
    models.test_appliances(test_windows, model_dir_name=model_dir_name)

    models.train_households(train_val_windows, full_train_windows, epochs=25, model_dir_name=model_dir_name)
    models.test_households(test_windows, model_dir_name=model_dir_name)

    models.train_communities(train_val_windows, full_train_windows, epochs=25, model_dir_name=model_dir_name)
    models.test_communities(test_windows, model_dir_name=model_dir_name)
    
    models.train_cities(train_val_windows, full_train_windows, epochs=25, model_dir_name=model_dir_name)
    models.test_cities(test_windows, model_dir_name=model_dir_name)

    result_utils = result_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX, model_dir_name=model_dir_name)
    result_utils.average_metrics_per_epoch()
    result_utils.summarize_evaluations()
    result_utils._average_appliance_evaluations()
    result_utils._average_household_evaluations()
    result_utils._average_community_evaluations()
    result_utils._average_city_evaluations()
    
def epochs_100_without_hyperparameters():
    # Split data
    train_val_windows, test_windows = models.split_data(TEST_FRACTION, 1)
    full_train_windows = train_val_windows[0][0]
    
    epochs = 100
    model_dir_name = "models_100_no_hp"

    models.train_households(train_val_windows, full_train_windows, epochs=epochs, model_dir_name=model_dir_name)
    models.test_households(test_windows, model_dir_name=model_dir_name)

    models.train_communities(train_val_windows, full_train_windows, epochs=epochs, model_dir_name=model_dir_name)
    models.test_communities(test_windows, model_dir_name=model_dir_name)
    
    models.train_cities(train_val_windows, full_train_windows, epochs=epochs, model_dir_name=model_dir_name)
    models.test_cities(test_windows, model_dir_name=model_dir_name)

    result_utils = result_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX, model_dir_name=model_dir_name)
    result_utils._average_metrics_per_epoch("/households/household=*/history_log_full.csv", "/households/", "household")
    result_utils._average_metrics_per_epoch("/communities/community=*/history_log_full.csv", "/communities/", "community")
    result_utils._average_metrics_per_epoch("/cities/city=*/history_log_full.csv", "/cities/", "city")
    result_utils.summarize_evaluations()
    result_utils._average_household_evaluations()
    result_utils._average_community_evaluations()
    result_utils._average_city_evaluations()
    
    
def epochs_25_with_hyperparameters():
    # Split data
    train_val_windows, test_windows = models.split_data(TEST_FRACTION, 1)
    full_train_windows = train_val_windows[0][0]
    
    # Initialize data_utils
    data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX)
    
    # Alternative model evaluation for community 1 only
    households = [661,1642,2335]
    semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS)
    epochs = 25
    model_dir_name = "models_25_hp"
    
    # Train appliances
    for household in households:
        sensors = [Process(target=models.learn,
                            args=(f"models/appliances/household={household}/appliance={sensor}/", model_dir_name,
                                    get_appliance_ec_input, get_appliance_ec_output, train_val_windows, full_train_windows,
                                    (household, sensor), semaphore, epochs, True)
                            ) for sensor in data_utils.get_appliances(CUSTOM_METADATA, household)]
        for sensor in sensors:
            sensor.start()
        for sensor in sensors:
            sensor.join()
    
    # Train households
    _households = [Process(target=models.learn,
                          args=(f"models/households/household={household}/", model_dir_name,
                                get_household_ec_input, get_household_ec_output, train_val_windows, full_train_windows,
                                (household,), semaphore, epochs, True)
                         ) for household in households]
    for household in _households:
        household.start()
    for household in _households:
        household.join()
        
    # Train community 1
    models.learn(f"models/communities/community=1/", model_dir_name, get_community_ec_input, get_community_ec_output, train_val_windows, full_train_windows, (1,), semaphore, epochs, True)
    
    # Aggregate forecasts
    hierarchy_community1 = data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA)['Austin'][1]
    hierarchy = {'Austin': {1: {household: hierarchy_community1[household] for household in households}}}
    
    appl_agg_layer = ApplianceAggregationLayer(hierarchy, model_dir_name)
    appl_agg_layer.from_disk()
    appl_agg_layer.evaluate(test_windows=test_windows)
    del appl_agg_layer
    
    household_agg_layer = HouseholdAggregationLayer(hierarchy, model_dir_name)
    household_agg_layer.from_disk()
    household_agg_layer.evaluate(test_windows=test_windows)
    del household_agg_layer
    
    result_utils = result_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX, model_dir_name=model_dir_name)
    
    result_utils._average_appliance_metrics_per_epoch()
    result_utils._average_metrics_per_epoch("/households/household=*/history_log_full.csv", "/households/", "household")
    result_utils.summarize_evaluations()
    result_utils._average_appliance_evaluations()
    result_utils._average_household_evaluations()
    
    
def save_expected_model_output(model_dir_name="models"):
    """
    Saves the expected model forecasts made on the test set to a txt file in the same format as the actual predictions in the `prediction.txt` files.
    Useful for manually comparing the expected vs the actual forecasts.
    """
    
    # Split data
    _, test_windows = models.split_data(TEST_FRACTION, 1)
    
    # Initialize data_utils
    data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX)
    
    # Get real energy consumption for verifying predictions
    for household in data_utils.get_households(CUSTOM_METADATA):
        ec = list(get_household_ec_output(SAVE_DIR, household, test_windows))
        np.savetxt(SAVE_DIR + f"/{model_dir_name}/households/household={household}/real.txt", ec)
        np.savetxt(SAVE_DIR + f"/{model_dir_name}/appliances/household={household}/real.txt", ec)
        for appliance in data_utils.get_appliances(CUSTOM_METADATA, household):
            np.savetxt(SAVE_DIR + f"/{model_dir_name}/appliances/household={household}/appliance={appliance}/real.txt", list(get_appliance_ec_output(SAVE_DIR, household, appliance, test_windows)))
    
    for community in data_utils.get_communities(CUSTOM_METADATA):
        ec = list(get_community_ec_output(SAVE_DIR, community, test_windows))
        np.savetxt(SAVE_DIR + f"/{model_dir_name}/communities/community={community}/real.txt", ec)
        np.savetxt(SAVE_DIR + f"/{model_dir_name}/households/community={community}/real.txt", ec)
        np.savetxt(SAVE_DIR + f"/{model_dir_name}/appliances/community={community}/real.txt", ec)
    
    for city in data_utils.get_cities(CUSTOM_METADATA):
        ec = list(get_city_ec_output(SAVE_DIR, city, test_windows))
        np.savetxt(SAVE_DIR + f"/{model_dir_name}/cities/city={city}/real.txt", ec)
        np.savetxt(SAVE_DIR + f"/{model_dir_name}/communities/city={city}/real.txt", ec)
        np.savetxt(SAVE_DIR + f"/{model_dir_name}/households/city={city}/real.txt", ec)
        np.savetxt(SAVE_DIR + f"/{model_dir_name}/appliances/city={city}/real.txt", ec)
        

def get_mean_ec():
    # Split data
    _, test_windows = models.split_data(TEST_FRACTION, 1)
    
    # Initialize data_utils
    data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX)
    
    ddf : dd.DataFrame = dd.read_parquet(SAVE_DIR + "/final_city/", columns=['index', 'total'])
    mean_ec_city = ddf[ddf['index'] >= test_windows[0]]['total'].mean().compute()
    
    num_cities = len(data_utils.get_cities(CUSTOM_METADATA))
    num_communities = len(data_utils.get_communities(CUSTOM_METADATA))
    households = data_utils.get_households(CUSTOM_METADATA)
    num_households = len(households)
    num_appliances = sum([len(data_utils.get_appliances(CUSTOM_METADATA, household)) for household in households])
    
    mean_ec_community = mean_ec_city * num_cities / num_communities
    mean_ec_household = mean_ec_city * num_cities / num_households
    mean_ec_appliance = mean_ec_city * num_cities / num_appliances
    
    return mean_ec_appliance, mean_ec_household, mean_ec_community, mean_ec_city

def get_std_ec():
    # Split data
    _, test_windows = models.split_data(TEST_FRACTION, 1)
    
    # Initialize data_utils
    data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX)
    
    ddf : dd.DataFrame = dd.read_parquet(SAVE_DIR + "/final_city/", columns=['index', 'total', 'city'])
    std_ec_city = ddf[ddf['index'] >= test_windows[0]].groupby(by=['city'])['total'].std().mean().compute()
    
    ddf : dd.DataFrame = dd.read_parquet(SAVE_DIR + "/final_community/", columns=['index', 'total', 'community'])
    std_ec_community = ddf[ddf['index'] >= test_windows[0]].groupby(by=['community'])['total'].std().mean().compute()
    
    ddf : dd.DataFrame = dd.read_parquet(SAVE_DIR + "/final_household/", columns=['index', 'total', 'dataid'])
    std_ec_household = ddf[ddf['index'] >= test_windows[0]].groupby(by=['dataid'])['total'].std().mean().compute()
    
    ddf : dd.DataFrame = dd.read_parquet(SAVE_DIR + "/final_appliance/", columns=['index', 'dataid']+[sensor for sensor in SENSOR_NAMES if sensor!="grid"])
    ddf = ddf[ddf['index'] >= test_windows[0]]
    del ddf['index']
    ddf[GENERATING_SENSOR_NAMES] = -1 * ddf[GENERATING_SENSOR_NAMES]
    
    std_total = 0
    total_appliances = 0
    for household in data_utils.get_households(CUSTOM_METADATA):
        appliances = data_utils.get_appliances(CUSTOM_METADATA, household)
        std_total += ddf[ddf['dataid']==household][appliances].std().sum().compute()
        total_appliances += len(appliances)
    std_ec_appliances = std_total / total_appliances
    
    return std_ec_appliances, std_ec_household, std_ec_community, std_ec_city
    
