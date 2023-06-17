import os
from math import floor
from multiprocessing import Process, Semaphore
from typing import List, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, losses, metrics, models, utils
from tensorflow import keras

from aggregation_layer import (ApplianceAggregationLayer, CityAggregationLayer,
                               CommunityAggregationLayer,
                               HouseholdAggregationLayer)
from constants import *
from data_utils import (data_utils_class, get_appliance_ec_input,
                        get_appliance_ec_output, get_city_ec_input,
                        get_city_ec_output, get_community_ec_input,
                        get_community_ec_output, get_household_ec_input,
                        get_household_ec_output)


def get_model():
    inp = layers.Input((INPUT_SIZE, 8))
    
    cnn = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(None ,INPUT_SIZE, 8))(inp)
    pool = layers.MaxPooling1D(pool_size=2)(cnn)

    # cnn_2 = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation="relu")(pool)
    # pool_2 = layers.MaxPooling1D(pool_size=2)(cnn_2)

    # lstm = layers.LSTM(units=64, activation="tanh", return_sequences=True)(pool)
    lstm_2 = layers.LSTM(units=64, activation="tanh")(pool)
    dense = layers.Dense(units=32)(lstm_2)
    out = layers.Dense(units=OUTPUT_SIZE)(dense)
    
    model = models.Model(inp, out)
    model.compile(loss='mae',
                optimizer='adam',
                metrics=[metrics.RootMeanSquaredError()])
    
    # utils.plot_model(model, show_shapes=True, show_layer_names=False)
    # print(model.summary())
    
    return model

def split_data(test_fraction: float, folds: int) -> Tuple[List[Tuple[List[int], List[int]]], List[int]]:
    r"""
    Splits the data in a test set and Kfold training and validation sets. Sets of data indices are converted to the windows they belong to, without creating overlap between the original sets.

    Parameters:
    ----------
    test_fraction : `float`
        The fraction of the data used for testing
    folds : `int`
        The number of kfold folds

    Returns:
    --------
    Tuple of kfolds and test window indices, where kfolds are represented as a list of tuples with train window indices and validation window indices.
        : `Tuple[List[Tuple[List[int], List[int]]], List[int]]`
    """
    date_range = pd.date_range(start=FROM_TIME, end=END_TIME, freq=TIME_FREQUENCY).to_frame()
    
    # windows = [*range(len(date_range) - input_window_size - output_window_size + 1)]
    test_windows = np.array([*range(floor((1 - test_fraction) * len(date_range)) + 1, len(date_range) + 2 - INPUT_SIZE - OUTPUT_SIZE)])
    if folds == 1:
        return [([*range(1, floor((1 - test_fraction) * len(date_range)) + 2 - INPUT_SIZE - OUTPUT_SIZE)], [])], test_windows
    else:
        fold_windows = np.array([np.asarray(fold)[:- INPUT_SIZE - OUTPUT_SIZE + 1] for fold in np.array_split([*range(1, floor((1 - test_fraction) * len(date_range)) + 1)], folds)])
        fold_indices = list(range(folds))
        return [(np.concatenate(fold_windows[[fold for fold in fold_indices if fold != i]]), fold_windows[i]) for i in fold_indices], test_windows


def compute_metrics(y_true, y_pred):
    rmse = metrics.RootMeanSquaredError()
    rmse.update_state(y_true, y_pred)
    return {'mae': losses.mae(y_true, y_pred), 'rmse': rmse.result().numpy()}
        

def learn(path, model_input, model_output, train_val_windows, full_train_windows, model_identifier, semaphore):
    os.makedirs(SAVE_DIR + path, exist_ok=True)
    
    # print("\n-------------------------------", f"|     Start KFold cross-validation for '{path}'...   |", "--------------------------------\n", flush=True)
    # folds = [Process(target=run_fold,
    #                     args=((SAVE_DIR, *model_identifier, train_val_windows[fold][0]),
    #                         (SAVE_DIR, *model_identifier, train_val_windows[fold][1]),
    #                         SAVE_DIR + path, model_input, model_output, fold, semaphore)
    #                     ) for fold in range(FOLDS)]
    # for fold in folds:
    #     fold.start()
    # for fold in folds:
    #     fold.join()
    
    # print("\n-------------------------------", f"|     Done with KFold cross-validation for '{path}'!    |", "--------------------------------\n", flush=True)

    with semaphore:
        model = get_model()
        model.fit(
                x=tf.data.Dataset.from_generator(generator=lambda *gen_args: ((inp, out) for inp, out in zip(model_input(*gen_args), model_output(*gen_args))),
                                                args=(SAVE_DIR, *model_identifier, full_train_windows),
                                                output_signature=(tf.TensorSpec(shape=(INPUT_SIZE, 8), dtype=tf.float32), tf.TensorSpec(shape=(OUTPUT_SIZE,), dtype=tf.float32))).batch(batch_size=BATCH_SIZE),
                epochs=EPOCHS,
                use_multiprocessing=True,
                callbacks=[tf.keras.callbacks.CSVLogger(SAVE_DIR+path + f"history_log_full.csv", separator=",")],
                max_queue_size=5
            )
        model.save(filepath=SAVE_DIR+path+"model/", overwrite=True)
            
    print("\n-------------------------------", f"|     Done with training for '{path}'!    |", "--------------------------------\n", flush=True)
    
def run_fold(train_args, val_args, path, model_input, model_output, fold, semaphore):
    with semaphore:
        gen = lambda *gen_args: ((inp, out) for inp, out in zip(model_input(*gen_args), model_output(*gen_args)))
        output_signature = (tf.TensorSpec(shape=(INPUT_SIZE, 8), dtype=tf.float32), tf.TensorSpec(shape=(OUTPUT_SIZE,), dtype=tf.float32))
        get_model().fit(
            x=tf.data.Dataset.from_generator(generator=gen,
                                            args=train_args,
                                            output_signature=output_signature).batch(batch_size=BATCH_SIZE),
            validation_data=tf.data.Dataset.from_generator(generator=gen,
                                            args=val_args,
                                            output_signature=output_signature).batch(batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            use_multiprocessing=True,
            callbacks=[tf.keras.callbacks.CSVLogger(path + f"history_log_fold_{fold}.csv", separator=",")],
            max_queue_size=5
        )

def predict(generator, generator_args, model, result_dict, key, semaphore):
    with semaphore:
        result_dict[key] = model.predict(x=tf.data.Dataset.from_generator(generator=generator,
                                                                          args=generator_args,
                                                                          output_signature=tf.TensorSpec(shape=(INPUT_SIZE, 8),
                                                                                                         dtype=tf.float32))
                                                          .batch(batch_size=BATCH_SIZE),
                                         use_multiprocessing=True)
        
def train_appliances(train_val_windows, full_train_windows, semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS), data_utils = data_utils_class()):
    print("\n-------------------------------", f"|     Start Training households...    |", "--------------------------------\n")

    for household in data_utils.get_households(CUSTOM_METADATA):
        sensors = [Process(target=learn,
                           args=(f"models/appliances/household={household}/appliance={sensor}/",
                                 get_appliance_ec_input, get_appliance_ec_output, train_val_windows, full_train_windows,
                                 (household, sensor), semaphore)
                          ) for sensor in data_utils.get_appliances(CUSTOM_METADATA, household)]
        for sensor in sensors:
            sensor.start()
        for sensor in sensors:
            sensor.join()
            
    print("\n-------------------------------", f"|     Done training appliances!    |", "--------------------------------\n")


def train_households(train_val_windows, full_train_windows, semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS), data_utils = data_utils_class()):
    print("\n-------------------------------", f"|     Start Training households...    |", "--------------------------------\n")

    households = [Process(target=learn,
                          args=(f"models/households/household={household}/",
                                get_household_ec_input, get_household_ec_output, train_val_windows, full_train_windows,
                                (household,), semaphore)
                         ) for household in data_utils.get_households(CUSTOM_METADATA)]
    for household in households:
        household.start()
    for household in households:
        household.join()
    
    print("\n-------------------------------", f"|     Done Training households!    |", "--------------------------------\n")


def train_communities(train_val_windows, full_train_windows, semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS), data_utils = data_utils_class()):
    print("\n-------------------------------", f"|     Start training communities...    |", "--------------------------------\n")

    communities = [Process(target=learn,
                          args=(f"models/communities/community={community}/",
                                get_community_ec_input, get_community_ec_output, train_val_windows, full_train_windows,
                                (community,), semaphore)
                         ) for community in data_utils.get_communities(CUSTOM_METADATA)]
    for community in communities:
        community.start()
    for community in communities:
        community.join()
        
    print("\n-------------------------------", f"|     Done training communities!    |", "--------------------------------\n")


def train_cities(train_val_windows, full_train_windows, semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS), data_utils = data_utils_class()):
    print("\n-------------------------------", f"|     Start training cities...    |", "--------------------------------\n")
    
    cities = [Process(target=models.learn,
                           args=(f"models/cities/city={city}/",
                                 get_city_ec_input, get_city_ec_output, train_val_windows, full_train_windows,
                                 (city,), semaphore)
                          ) for city in data_utils.get_cities(CUSTOM_METADATA)]
    for city in cities:
        city.start()
    for city in cities:
        city.join()
    
    print("\n-------------------------------", f"|     Done training cities!    |", "--------------------------------\n")


def test_appliances(test_windows, data_utils = data_utils_class()):
    print("\n-------------------------------", f"|     Start testing appliances...    |", "--------------------------------\n")

    # Make appliance predictions and aggregate them to all levels above while recording predictions and metrics
    appl_agg_layer = ApplianceAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA))
    appl_agg_layer.from_disk()
    appl_agg_layer.evaluate(test_windows=test_windows)
    print("\n-------------------------------", f"|     Done testing appliances!    |", "--------------------------------\n")


def test_households(test_windows, data_utils = data_utils_class()):
    print("\n-------------------------------", f"|     Start testing households...    |", "--------------------------------\n")
    # Make household predictions and aggregate them to all levels above while recording predictions and metrics
    house_agg_layer = HouseholdAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA))
    house_agg_layer.from_disk()
    house_agg_layer.evaluate(test_windows=test_windows)

    print("\n-------------------------------", f"|     Done testing households!    |", "--------------------------------\n")
    
def test_communities(test_windows, data_utils = data_utils_class()):
    print("\n-------------------------------", f"|     Start testing communities...    |", "--------------------------------\n")

    # Make community predictions and aggregate them to all levels above while recording predictions and metrics
    comm_agg_layer = CommunityAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA))
    comm_agg_layer.from_disk()
    comm_agg_layer.evaluate(test_windows=test_windows)
    
    print("\n-------------------------------", f"|     Done testing communities!    |", "--------------------------------\n")
    
    
def test_cities(test_windows, data_utils = data_utils_class()):
    print("\n-------------------------------", f"|     Start testing cities...    |", "--------------------------------\n")

    # Make city predictions and aggregate them to all levels above while recording predictions and metrics
    city_agg_layer = CityAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA))
    city_agg_layer.from_disk()
    city_agg_layer.evaluate(test_windows=test_windows)
    
    print("\n-------------------------------", f"|     Done testing cities!    |", "--------------------------------\n")
    
    