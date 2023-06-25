import os
from functools import reduce
from math import floor
from multiprocessing import Process, Semaphore
from typing import List, Tuple

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, metrics, models
from sklearn.model_selection import train_test_split

from aggregation_layer import (ApplianceAggregationLayer, CityAggregationLayer,
                               CommunityAggregationLayer,
                               HouseholdAggregationLayer)
from constants import *
from data_utils import (data_utils_class, get_appliance_ec_input,
                        get_appliance_ec_output, get_city_ec_input,
                        get_city_ec_output, get_community_ec_input,
                        get_community_ec_output, get_household_ec_input,
                        get_household_ec_output)


def get_model(cnn_layers: int = 1, cnn_filters: int = 64, lstm_layers: int = 1, lstm_units: int = 64, dense_units: int = 32, dropout: bool = False):
    inp = layers.Input((INPUT_SIZE, 8))
    
    cnn = get_cnn_layers(cnn_layers, inp, cnn_filters)
    dropout_1 = layers.Dropout(0.25)(cnn) if dropout else cnn
    lstm = get_lstm_layers(lstm_layers, dropout_1, lstm_units)
    dropout_2 = layers.Dropout(0.25)(lstm) if dropout else lstm
    
    dense = layers.Dense(units=dense_units)(dropout_2)
    out = layers.Dense(units=OUTPUT_SIZE)(dense)
    
    model = models.Model(inp, out)
    model.compile(loss='mae',
                optimizer='adam',
                metrics=[metrics.RootMeanSquaredError()])
    
    return model
    
def get_model_hp(hp: kt.HyperParameters):
    inp = layers.Input((INPUT_SIZE, 8))
    
    cnn = get_cnn_layers(hp.Int('cnn_layers', min_value=1, max_value=3, step=1), inp, hp.Int('cnn_filters', min_value=16, max_value=128, step=2, sampling="log"))
    dropout_1 = layers.Dropout(0.25)(cnn)
    lstm = get_lstm_layers(hp.Int('lstm_layers', min_value=1, max_value=3, step=1), dropout_1, hp.Int('lstm_units', min_value=16, max_value=128, step=2, sampling="log"))
    dropout_2 = layers.Dropout(0.25)(lstm)
    
    dense = layers.Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=2, sampling="log"))(dropout_2)
    out = layers.Dense(units=OUTPUT_SIZE)(dense)
    
    model = models.Model(inp, out)
    model.compile(loss='mae',
                optimizer='adam',
                metrics=[metrics.RootMeanSquaredError()])
    
    # utils.plot_model(model, show_shapes=True, show_layer_names=False)
    # print(model.summary())
    
    return model

def get_cnn_layers(n_layers: int, inp, filters: int):
    if (n_layers < 1):
        return inp
    cnn = layers.Conv1D(filters=filters, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(None ,INPUT_SIZE, 8))(inp)
    pooling = layers.MaxPooling1D(pool_size=2)(cnn)
    for i in range(n_layers-1):
        cnn = layers.Conv1D(filters=filters / (2**(i+1)), kernel_size=2, strides=1, padding='same', activation='relu')(pooling)
        pooling = layers.MaxPooling1D(pool_size=2)(cnn)
        
    return pooling

def get_lstm_layers(n_layers: int, inp, units):
    if (n_layers < 1):
        return inp
    first_layers = reduce(lambda x, _: layers.LSTM(units=units, activation="tanh", return_sequences=True)(x), range(n_layers-1), inp)
    return layers.LSTM(units=units, activation="tanh")(first_layers)

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


def learn(path, model_dir_name, model_input, model_output, train_val_windows, full_train_windows, model_identifier, semaphore, epochs, hp_optimization: bool):
    os.makedirs(SAVE_DIR +"/" + model_dir_name + "/" + path, exist_ok=True)

    with semaphore:
        model = None
        if (hp_optimization):
            # Optimize HyperParameters
            train_windows, val_windows = train_test_split(full_train_windows, test_size=VAL_FRACTION)

            tuner = kt.Hyperband(get_model_hp, objective='val_loss', max_epochs=10, factor=3, overwrite=True, directory=SAVE_DIR+"/hyper_parameters/", project_name=path)
            tuner.search(tf.data.Dataset.from_generator(generator=lambda *gen_args: ((inp, out) for inp, out in zip(model_input(*gen_args), model_output(*gen_args))),
                                                        args=(SAVE_DIR, *model_identifier, train_windows),
                                                        output_signature=(tf.TensorSpec(shape=(INPUT_SIZE, 8), dtype=tf.float32), tf.TensorSpec(shape=(OUTPUT_SIZE,), dtype=tf.float32))).batch(batch_size=BATCH_SIZE),
                        validation_data = tf.data.Dataset.from_generator(generator=lambda *gen_args: ((inp, out) for inp, out in zip(model_input(*gen_args), model_output(*gen_args))),
                                                                         args=(SAVE_DIR, *model_identifier, val_windows),
                                                                         output_signature=(tf.TensorSpec(shape=(INPUT_SIZE, 8), dtype=tf.float32), tf.TensorSpec(shape=(OUTPUT_SIZE,), dtype=tf.float32))).batch(batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        use_multiprocessing=True,
                        callbacks=[tf.keras.callbacks.CSVLogger(f"{SAVE_DIR}/{model_dir_name}/{path}/history_log_full.csv", separator=",")],
                        max_queue_size=5
                        )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps)
        else:
            model = get_model()
        

        model.fit(
                x=tf.data.Dataset.from_generator(generator=lambda *gen_args: ((inp, out) for inp, out in zip(model_input(*gen_args), model_output(*gen_args))),
                                                args=(SAVE_DIR, *model_identifier, full_train_windows),
                                                output_signature=(tf.TensorSpec(shape=(INPUT_SIZE, 8), dtype=tf.float32), tf.TensorSpec(shape=(OUTPUT_SIZE,), dtype=tf.float32))).batch(batch_size=BATCH_SIZE),
                epochs=EPOCHS,
                use_multiprocessing=True,
                callbacks=[tf.keras.callbacks.CSVLogger(f"{SAVE_DIR}/{model_dir_name}/{path}/history_log_full.csv", separator=",")],
                max_queue_size=5
            )
        model.save(filepath=f"{SAVE_DIR}/{model_dir_name}/{path}/model/", overwrite=True)
            
    print("\n-------------------------------", f"|     Done with training for '{path}'!    |", "--------------------------------\n", flush=True)

        
def train_appliances(train_val_windows, full_train_windows, semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS), data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX), hp_optimization = False, epochs=EPOCHS, model_dir_name="models"):
    print("\n-------------------------------", f"|     Start Training households...    |", "--------------------------------\n")

    for household in data_utils.get_households(CUSTOM_METADATA):
        sensors = [Process(target=learn,
                           args=(f"/appliances/household={household}/appliance={sensor}/", model_dir_name,
                                 get_appliance_ec_input, get_appliance_ec_output, train_val_windows, full_train_windows,
                                 (household, sensor), semaphore, epochs, hp_optimization)
                          ) for sensor in data_utils.get_appliances(CUSTOM_METADATA, household)]
        for sensor in sensors:
            sensor.start()
        for sensor in sensors:
            sensor.join()
            
    print("\n-------------------------------", f"|     Done training appliances!    |", "--------------------------------\n")


def train_households(train_val_windows, full_train_windows, semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS), data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX), hp_optimization = False, epochs=EPOCHS, model_dir_name="models"):
    print("\n-------------------------------", f"|     Start Training households...    |", "--------------------------------\n")

    households = [Process(target=learn,
                          args=(f"{model_dir_name}/households/household={household}/", model_dir_name,
                                get_household_ec_input, get_household_ec_output, train_val_windows, full_train_windows,
                                (household,), semaphore, epochs, hp_optimization)
                         ) for household in data_utils.get_households(CUSTOM_METADATA)]
    for household in households:
        household.start()
    for household in households:
        household.join()
    
    print("\n-------------------------------", f"|     Done Training households!    |", "--------------------------------\n")


def train_communities(train_val_windows, full_train_windows, semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS), data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX), hp_optimization = False, epochs=EPOCHS, model_dir_name="models"):
    print("\n-------------------------------", f"|     Start training communities...    |", "--------------------------------\n")

    communities = [Process(target=learn,
                          args=(f"{model_dir_name}/communities/community={community}/", model_dir_name,
                                get_community_ec_input, get_community_ec_output, train_val_windows, full_train_windows,
                                (community,), semaphore, epochs, hp_optimization)
                         ) for community in data_utils.get_communities(CUSTOM_METADATA)]
    for community in communities:
        community.start()
    for community in communities:
        community.join()
        
    print("\n-------------------------------", f"|     Done training communities!    |", "--------------------------------\n")


def train_cities(train_val_windows, full_train_windows, semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS), data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX), hp_optimization = False, epochs=EPOCHS, model_dir_name="models"):
    print("\n-------------------------------", f"|     Start training cities...    |", "--------------------------------\n")
    
    cities = [Process(target=learn,
                           args=(f"{model_dir_name}/cities/city={city}/", model_dir_name,
                                 get_city_ec_input, get_city_ec_output, train_val_windows, full_train_windows,
                                 (city,), semaphore, epochs, hp_optimization)
                          ) for city in data_utils.get_cities(CUSTOM_METADATA)]
    for city in cities:
        city.start()
    for city in cities:
        city.join()
    
    print("\n-------------------------------", f"|     Done training cities!    |", "--------------------------------\n")


def test_appliances(test_windows, data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX), model_dir_name="models"):
    print("\n-------------------------------", f"|     Start testing appliances...    |", "--------------------------------\n")

    # Make appliance predictions and aggregate them to all levels above while recording predictions and metrics
    appl_agg_layer = ApplianceAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA), model_dir_name)
    # appl_agg_layer.from_disk()
    # appl_agg_layer.evaluate(test_windows=test_windows)
    appl_agg_layer.from_predictions(test_windows)
    print("\n-------------------------------", f"|     Done testing appliances!    |", "--------------------------------\n")


def test_households(test_windows, data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX), model_dir_name="models"):
    print("\n-------------------------------", f"|     Start testing households...    |", "--------------------------------\n")
    # Make household predictions and aggregate them to all levels above while recording predictions and metrics
    house_agg_layer = HouseholdAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA), model_dir_name)
    # house_agg_layer.from_disk()
    # house_agg_layer.evaluate(test_windows=test_windows)
    house_agg_layer.from_predictions(test_windows)

    print("\n-------------------------------", f"|     Done testing households!    |", "--------------------------------\n")
    
def test_communities(test_windows, data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX), model_dir_name="models"):
    print("\n-------------------------------", f"|     Start testing communities...    |", "--------------------------------\n")

    # Make community predictions and aggregate them to all levels above while recording predictions and metrics
    comm_agg_layer = CommunityAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA), model_dir_name)
    # comm_agg_layer.from_disk()
    # comm_agg_layer.evaluate(test_windows=test_windows)
    comm_agg_layer.from_predictions(test_windows)
    
    print("\n-------------------------------", f"|     Done testing communities!    |", "--------------------------------\n")
    
    
def test_cities(test_windows, data_utils = data_utils_class(time_column_name=TIME_COLUMN_NAME, time_frequency=TIME_FREQUENCY, metadata_time_prefix=METADATA_TIME_PREFIX), model_dir_name="models"):
    print("\n-------------------------------", f"|     Start testing cities...    |", "--------------------------------\n")

    # Make city predictions and aggregate them to all levels above while recording predictions and metrics
    city_agg_layer = CityAggregationLayer(data_utils.get_hierarchy_dict(metadata_files=CUSTOM_METADATA), model_dir_name)
    # city_agg_layer.from_disk()
    # city_agg_layer.evaluate(test_windows=test_windows)
    city_agg_layer.from_predictions(test_windows)
    print("\n-------------------------------", f"|     Done testing cities!    |", "--------------------------------\n")
    
    