from multiprocessing import Process, Semaphore, Manager
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, metrics, losses, utils
import dask.dataframe as dd
import pandas as pd
from constants import *
from typing import List, Tuple
import numpy as np
from math import floor
import os


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


# Generators for model input and expected output
################################################################################################################

def get_appliance_ec_input(data_path: str, dataid: int, sensor: str, windows: List[int]):
    if type(data_path) == bytes:
        data_path = data_path.decode('utf-8')
    if type(sensor) == bytes:
        sensor = sensor.decode('utf-8')
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/normalized/final_appliance/dataid="+str(dataid), columns=[sensor, 'index'] + FEATURE_COLUMNS)
    for window in windows:
        indexes = [*range(window, window + INPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][[sensor] + FEATURE_COLUMNS].compute().to_numpy(dtype=np.float64)
        yield np_array

def get_appliance_ec_output(data_path: str, dataid: int, sensor: str, windows: List[int]):
    if type(data_path) == bytes:
        data_path = data_path.decode('utf-8')
    if type(sensor) == bytes:
        sensor = sensor.decode('utf-8')
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_appliance/dataid="+str(dataid), columns=[sensor, 'index'] + FEATURE_COLUMNS)
    for window in windows:
        indexes = [*range(window + INPUT_SIZE, window + INPUT_SIZE + OUTPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][sensor].compute().to_numpy(dtype=np.float64)
        yield np_array
        
def get_household_ec_input(data_path: str, dataid: int, windows: List[int]):
    if type(data_path) == bytes:
        data_path = data_path.decode('utf-8')
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/normalized/final_household/dataid="+str(dataid), columns=['index', 'total'] + FEATURE_COLUMNS)
    for window in windows:
        indexes = [*range(window, window + INPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][['total'] + FEATURE_COLUMNS].compute().to_numpy(dtype=np.float64)
        yield np_array

def get_household_ec_output(data_path: str, dataid: int, windows: List[int]):
    if type(data_path) == bytes:
        data_path = data_path.decode('utf-8')
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_household/dataid="+str(dataid), columns=['index', 'total'] + FEATURE_COLUMNS)
    for window in windows:
        indexes = [*range(window + INPUT_SIZE, window + INPUT_SIZE + OUTPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)]['total'].compute().to_numpy(dtype=np.float64)
        yield np_array
        
def get_community_ec_input(data_path: str, community: int, windows: List[int]):
    if type(data_path) == bytes:
        data_path = data_path.decode('utf-8')
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/normalized/final_community/community="+str(community), columns=['index', 'total'] + FEATURE_COLUMNS)
    for window in windows:
        indexes = [*range(window, window + INPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][['total'] + FEATURE_COLUMNS].compute().to_numpy(dtype=np.float64)
        yield np_array

def get_community_ec_output(data_path: str, community: int, windows: List[int]):
    if type(data_path) == bytes:
        data_path = data_path.decode('utf-8')
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_community/community="+str(community), columns=['index', 'total'] + FEATURE_COLUMNS)
    for window in windows:
        indexes = [*range(window + INPUT_SIZE, window + INPUT_SIZE + OUTPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)]['total'].compute().to_numpy(dtype=np.float64)
        yield np_array
        
def get_city_ec_input(data_path: str, city: str, windows: List[int]):
    if type(data_path) == bytes:
        data_path = data_path.decode('utf-8')
    if type(city) == bytes:
        city = city.decode('utf-8')
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/normalized/final_city/city="+city, columns=['index', 'total'] + FEATURE_COLUMNS)
    for window in windows:
        indexes = [*range(window, window + INPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][['total'] + FEATURE_COLUMNS].compute().to_numpy(dtype=np.float64)
        yield np_array

def get_city_ec_output(data_path: str, city: str, windows: List[int]):
    if type(data_path) == bytes:
        data_path = data_path.decode('utf-8')
    if type(city) == bytes:
        city = city.decode('utf-8')
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_city/city="+city, columns=['index', 'total'] + FEATURE_COLUMNS)
    for window in windows:
        indexes = [*range(window + INPUT_SIZE, window + INPUT_SIZE + OUTPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)]['total'].compute().to_numpy(dtype=np.float64)
        yield np_array
        
######################################################################################

def compute_metrics(y_true, y_pred):
    rmse = metrics.RootMeanSquaredError()
    rmse.update_state(y_true, y_pred)
    # return {'mae': losses.mae(y_true, y_pred), 'rmse': rmse.result().numpy(), 'mse': losses.MSE(y_true, y_pred), 'mape': losses.mape(y_true, y_pred)}
    return {'mae': losses.mae(y_true, y_pred), 'rmse': rmse.result().numpy()}
        
class ApplianceAggregationLayer:
    def __init__(self, hierarchy : dict) -> None:
        self.hierarchy = hierarchy
        self.data_path = SAVE_DIR
        
    def add_model(self, city, community, household, appliance, model):
        self.hierarchy[city][community][household][appliance] = model
        
    def from_disk(self):
        for city in self.hierarchy:
            for community in self.hierarchy[city]:
                for household in self.hierarchy[city][community]:
                    for appliance in self.hierarchy[city][community][household]:
                        self.hierarchy[city][community][household][appliance] = models.load_model(self.data_path + f"models/appliances/household={household}/appliance={appliance}/model")
                        
    def evaluate(self, test_windows : List[int]):
        with Manager() as manager:
            semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS)
            processes = []
            for city, communities in self.hierarchy.items():
                for community, households in communities.items():
                    for household, appliances in households.items():
                        result_dict = manager.dict()
                        for appliance, model in appliances.items():
                            process = Process(target=predict,
                                              args=(get_appliance_ec_input,
                                                    (self.data_path, household, appliance, test_windows),
                                                    model, result_dict, appliance, semaphore))
                            process.start()
                            processes.append(process)
                        self.hierarchy[city][community][household] = result_dict

            for process in processes:
                process.join()
            for city, communities in self.hierarchy.items():
                community_predictions = []
                for community, households in communities.items():
                    household_predictions = []
                    for household, appliances in households.items():
                        appliance_predictions = []
                        for appliance, appliance_prediction in appliances.items():
                            np.savetxt(self.data_path + f"/models/appliances/household={household}/appliance={appliance}/prediction.txt", appliance_prediction)
                            pd.DataFrame.from_dict(compute_metrics(list(get_appliance_ec_output(self.data_path, household, appliance, test_windows)), appliance_prediction)).to_csv(self.data_path+f"/models/appliances/household={household}/appliance={appliance}/evaluation.csv", index=False)
                            appliance_predictions.append(appliance_prediction)
                        household_prediction = np.add.reduce(appliance_predictions)
                        os.makedirs(f"/models/appliances/household={household}/", exist_ok=True)
                        np.savetxt(self.data_path + f"/models/appliances/household={household}/prediction.txt", household_prediction)
                        pd.DataFrame.from_dict(compute_metrics(list(get_household_ec_output(self.data_path, household, test_windows)), household_prediction)).to_csv(self.data_path+f"/models/appliances/household={household}/evaluation.csv", index=False)
                        household_predictions.append(household_prediction)
                    community_prediction = np.add.reduce(household_predictions)
                    os.makedirs(f"/models/appliances/community={city}/", exist_ok=True)
                    np.savetxt(self.data_path + f"/models/appliances/community={community}/prediction.txt", community_prediction)
                    pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(self.data_path+f"/models/appliances/community={community}/evaluation.csv", index=False)
                    community_predictions.append(community_prediction)
                city_prediction = np.add.reduce(community_predictions)
                os.makedirs(f"/models/appliances/city={city}/", exist_ok=True)
                np.savetxt(self.data_path + f"/models/appliances/city={city}/prediction.txt", city_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(self.data_path+f"/models/appliances/city={city}/evaluation.csv", index=False)


class HouseholdAggregationLayer:
    def __init__(self, hierarchy : dict) -> None:
        self.hierarchy = hierarchy
        self.data_path = SAVE_DIR
        
    def add_model(self, city, community, household, model):
        self.hierarchy[city][community][household] = model
        
    def from_disk(self):
        for city in self.hierarchy:
            for community in self.hierarchy[city]:
                for household in self.hierarchy[city][community]:
                    self.hierarchy[city][community][household] = models.load_model(self.data_path + f"models/households/household={household}/model")
                        
    def evaluate(self, test_windows : List[int]):
        with Manager() as manager:
            semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS)
            processes = []
            for city, communities in self.hierarchy.items():
                for community, households in communities.items():
                    result_dict = manager.dict()
                    for household, model in households.items():
                        process = Process(target=predict,
                                          args=(get_household_ec_input,
                                                (self.data_path, household, test_windows),
                                                model, result_dict, household, semaphore))
                        process.start()
                        processes.append(process)
                    self.hierarchy[city][community] = result_dict

            for process in processes:
                process.join()
            for city, communities in self.hierarchy.items():
                community_predictions = []
                for community, households in communities.items():
                    household_predictions = []
                    for household, household_prediction in households.items():
                        np.savetxt(self.data_path + f"/models/households/household={household}/prediction.txt", household_prediction)
                        pd.DataFrame.from_dict(compute_metrics(list(get_household_ec_output(self.data_path, household, test_windows)), household_prediction)).to_csv(self.data_path+f"/models/households/household={household}/evaluation.csv", index=False)
                        household_predictions.append(household_prediction)
                    community_prediction = np.add.reduce(household_predictions)
                    os.makedirs(f"/models/households/community={community}/", exist_ok=True)
                    np.savetxt(self.data_path + f"/models/households/community={community}/prediction.txt", community_prediction)
                    pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(self.data_path+f"/models/households/community={community}/evaluation.csv", index=False)
                    community_predictions.append(community_prediction)
                city_prediction = np.add.reduce(community_predictions)
                os.makedirs(f"/models/households/city={city}/", exist_ok=True)
                np.savetxt(self.data_path + f"/models/households/city={city}/prediction.txt", city_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(self.data_path+f"/models/households/city={city}/evaluation.csv", index=False)


class CommunityAggregationLayer:
    def __init__(self, hierarchy : dict) -> None:
        self.hierarchy = hierarchy
        self.data_path = SAVE_DIR
        
    def add_model(self, city, community, model):
        self.hierarchy[city][community] = model
        
    def from_disk(self):
        for city in self.hierarchy:
            for community in self.hierarchy[city]:
                self.hierarchy[city][community] = models.load_model(self.data_path + f"models/communities/community={community}/model")
                        
    def evaluate(self, test_windows : List[int]):
        with Manager() as manager:
            semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS)
            processes = []
            for city, communities in self.hierarchy.items():
                result_dict = manager.dict()
                for community, model in communities.items():
                    process = Process(target=predict,
                                      args=(get_community_ec_input,
                                            (self.data_path, community, test_windows),
                                            model, result_dict, community, semaphore))
                    process.start()
                    processes.append(process)
                self.hierarchy[city] = result_dict
            for process in processes:
                process.join()
            for city, communities in self.hierarchy.items():
                community_predictions = []
                for community, community_prediction in communities.items():
                    np.savetxt(self.data_path + f"/models/communities/community={community}/prediction.txt", community_prediction)
                    pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(self.data_path+f"/models/communities/community={community}/evaluation.csv", index=False)
                    community_predictions.append(community_prediction)
                city_prediction = np.add.reduce(community_predictions)
                os.makedirs(f"/models/communities/city={city}/", exist_ok=True)
                np.savetxt(self.data_path + f"/models/communities/city={city}/prediction.txt", city_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(self.data_path+f"/models/communities/city={city}/evaluation.csv", index=False)
                    

class CityAggregationLayer:
    def __init__(self, hierarchy : dict) -> None:
        self.hierarchy = hierarchy
        self.data_path = SAVE_DIR
        
    def add_model(self, city, model):
        self.hierarchy[city] = model
        
    def from_disk(self):
        for city in self.hierarchy:
            self.hierarchy[city] = models.load_model(self.data_path + f"models/cities/city={city}/model")
                        
    def evaluate(self, test_windows : List[int]):
        with Manager() as manager:
            semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS)
            processes = []
            result_dict = manager.dict()
            for city, model in self.hierarchy.items():
                process = Process(target=predict,
                                  args=(get_city_ec_input,
                                        (self.data_path, community, test_windows),
                                        model, result_dict, community, semaphore))
                process.start()
                processes.append(process)
            self.hierarchy[city] = result_dict
            for process in processes:
                process.join()
            for city, city_prediction in self.hierarchy.items():
                np.savetxt(self.data_path + f"/models/cities/city={city}/prediction.txt", city_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(self.data_path+f"/models/cities/city={city}/evaluation.csv", index=False)

def learn(path, model_input, model_output, train_val_windows, full_train_windows, model_identifier, semaphore):
    # print("\n-------------------------------", f"|     Start KFold cross-validation for '{path}'...   |", "--------------------------------\n", flush=True)
    os.makedirs(SAVE_DIR + path, exist_ok=True)
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