import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, metrics, losses, utils
import dask.dataframe as dd
import pandas as pd
from constants import FEATURE_COLUMNS, INPUT_SIZE, OUTPUT_SIZE, SAVE_DIR, TIME_FREQUENCY, FROM_TIME, END_TIME
from typing import List, Tuple
import numpy as np
from math import floor


def get_model():
    inp = layers.Input((INPUT_SIZE, 8), ragged=True)

    cnn = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(INPUT_SIZE, 8))(inp)
    pool = layers.MaxPool1D(pool_size=2)(cnn)

    cnn_2 = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation="relu")(pool)
    pool_2 = layers.MaxPooling1D(pool_size=2)(cnn_2)

    lstm = layers.LSTM(units=64, activation="tanh", return_sequences=True)(pool_2)
    lstm_2 = layers.LSTM(units=64, activation="tanh")(lstm)
    dense = layers.Dense(units=32)(lstm_2)
    out = layers.Dense(units=OUTPUT_SIZE)(dense)

    model = models.Model(inp, out)
    model.compile(loss='mae',
                optimizer='adam',
                metrics=['mae', 'rmse', 'mse', 'mape'])
    
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
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_appliance", columns=[sensor, 'index', 'dataid'] + FEATURE_COLUMNS)
    ddf = ddf[ddf['dataid'] == dataid]
    for window in windows:
        indexes = [*range(window, window + INPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][[sensor] + FEATURE_COLUMNS].values.compute()
        yield np_array

def get_appliance_ec_output(data_path: str, dataid: int, sensor: str, windows: List[int]):
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_appliance", columns=[sensor, 'index', 'dataid'] + FEATURE_COLUMNS)
    ddf = ddf[ddf['dataid'] == dataid]
    for window in windows:
        indexes = [*range(window + INPUT_SIZE, window + INPUT_SIZE + OUTPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][sensor].values.compute()
        yield np_array
        
def get_household_ec_input(data_path: str, dataid: int, windows: List[int]):
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_household", columns=['index', 'dataid', 'total'] + FEATURE_COLUMNS)
    ddf = ddf[ddf['dataid'] == dataid]
    for window in windows:
        indexes = [*range(window, window + INPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][['total'] + FEATURE_COLUMNS].values.compute()
        yield np_array

def get_household_ec_output(data_path: str, dataid: int, windows: List[int]):
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_household", columns=['index', 'dataid', 'total'] + FEATURE_COLUMNS)
    ddf = ddf[ddf['dataid'] == dataid]
    for window in windows:
        indexes = [*range(window + INPUT_SIZE, window + INPUT_SIZE + OUTPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)]['total'].values.compute()
        yield np_array
        
def get_community_ec_input(data_path: str, community: int, windows: List[int]):
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_household", columns=['index', 'community', 'total'] + FEATURE_COLUMNS)
    ddf = ddf[ddf['community'] == community]
    for window in windows:
        indexes = [*range(window, window + INPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][['total'] + FEATURE_COLUMNS].values.compute()
        yield np_array

def get_community_ec_output(data_path: str, community: int, windows: List[int]):
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_household", columns=['index', 'community', 'total'] + FEATURE_COLUMNS)
    ddf = ddf[ddf['community'] == community]
    for window in windows:
        indexes = [*range(window + INPUT_SIZE, window + INPUT_SIZE + OUTPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)]['total'].values.compute()
        yield np_array
        
def get_city_ec_input(data_path: str, city: str, windows: List[int]):
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_household", columns=['index', 'city', 'total'] + FEATURE_COLUMNS)
    ddf = ddf[ddf['community'] == city]
    for window in windows:
        indexes = [*range(window, window + INPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)][['total'] + FEATURE_COLUMNS].values.compute()
        yield np_array

def get_city_ec_output(data_path: str, city: str, windows: List[int]):
    ddf : dd.DataFrame = dd.read_parquet(data_path+"/final_household", columns=['index', 'city', 'total'] + FEATURE_COLUMNS)
    ddf = ddf[ddf['community'] == city]
    for window in windows:
        indexes = [*range(window + INPUT_SIZE, window + INPUT_SIZE + OUTPUT_SIZE)]
        np_array = ddf[ddf['index'].isin(indexes)]['total'].values.compute()
        yield np_array
        
######################################################################################

def compute_metrics(y_true, y_pred):
    rmse = metrics.RootMeanSquaredError()
    rmse.update_state(y_true, y_pred)
    return {'mae': losses.mae(y_true, y_pred), 'rmse': rmse.result().numpy(), 'mse': losses.MSE(y_true, y_pred), 'mape': losses.mape(y_true, y_pred)}

        
        
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
        for city in self.hierarchy:
            for community in self.hierarchy[city]:
                for household in self.hierarchy[city][community]:
                    for appliance in self.hierarchy[city][community][household]:
                        self.hierarchy[city][community][household][appliance] = self.hierarchy[city][community][household][appliance].predict(
                            x=tf.data.Dataset.from_generator(
                                generator=get_appliance_ec_input,
                                args=(self.data_path, household, appliance, test_windows),
                                output_signature=tf.RaggedTensorSpec(shape=(INPUT_SIZE, 8, 1), dtype=tf.float32)).batch(batch_size=1).batch(batch_size=1),
                            use_multiprocessing=True                                                                                                                  
                        )
                        np.savetxt(self.data_path + f"/models/appliances/household={household}/appliance={appliance}/prediction.txt", self.hierarchy[city][community][household][appliance])
                        pd.from_dict(compute_metrics(list(get_appliance_ec_output(self.data_path, household, appliance, test_windows)), self.hierarchy[city][community][household][appliance])).to_csv(self.data_path+f"/models/appliances/household={household}/appliance={appliance}/evaluation.csv", index=False)
                    self.hierarchy[city][community][household] = np.add.reduce(self.hierarchy[city][community][household].values())
                    np.savetxt(self.data_path + f"/models/appliances/household={household}/prediction.txt", self.hierarchy[city][community][household])
                    pd.from_dict(compute_metrics(list(get_household_ec_output(self.data_path, household, test_windows)), self.hierarchy[city][community][household])).to_csv(self.data_path+f"/models/appliances/household={household}/evaluation.csv", index=False)
                self.hierarchy[city][community] = np.add.reduce(self.hierarchy[city][community].values())
                np.savetxt(self.data_path + f"/models/appliances/community={community}/prediction.txt", self.hierarchy[city][community])
                pd.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), self.hierarchy[city][community])).to_csv(self.data_path+f"/models/appliances/community={community}/evaluation.csv", index=False)
            self.hierarchy[city] = np.add.reduce(self.hierarchy[city].values())
            np.savetxt(self.data_path + f"/models/appliances/city={community}/prediction.txt", self.hierarchy[city])
            pd.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), self.hierarchy[city])).to_csv(self.data_path+f"/models/appliances/city={city}/evaluation.csv", index=False)
                    

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
        for city in self.hierarchy:
            for community in self.hierarchy[city]:
                for household in self.hierarchy[city][community]:
                    self.hierarchy[city][community][household] = self.hierarchy[city][community][household].predict(
                        x=tf.data.Dataset.from_generator(
                            generator=get_household_ec_input,
                            args=(self.data_path, household, test_windows),
                            output_signature=tf.RaggedTensorSpec(shape=(INPUT_SIZE, 8, 1), dtype=tf.float32)).batch(batch_size=1).batch(batch_size=1),
                        use_multiprocessing=True                                                                                                                  
                    )
                    np.savetxt(self.data_path + f"/models/households/household={household}/prediction.txt", self.hierarchy[city][community][household])
                    pd.from_dict(compute_metrics(list(get_household_ec_output(self.data_path, household, test_windows)), self.hierarchy[city][community][household])).to_csv(self.data_path+f"/models/households/household={household}/evaluation.csv", index=False)
                self.hierarchy[city][community] = np.add.reduce(self.hierarchy[city][community].values())
                np.savetxt(self.data_path + f"/models/households/community={community}/prediction.txt", self.hierarchy[city][community])
                pd.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), self.hierarchy[city][community])).to_csv(self.data_path+f"/models/households/community={community}/evaluation.csv", index=False)
            self.hierarchy[city] = np.add.reduce(self.hierarchy[city].values())
            np.savetxt(self.data_path + f"/models/households/city={community}/prediction.txt", self.hierarchy[city])
            pd.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), self.hierarchy[city])).to_csv(self.data_path+f"/models/households/city={city}/evaluation.csv", index=False)
                    

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
        for city in self.hierarchy:
            for community in self.hierarchy[city]:
                self.hierarchy[city][community] = self.hierarchy[city][community].predict(
                    x=tf.data.Dataset.from_generator(
                        generator=get_community_ec_input,
                        args=(self.data_path, community, test_windows),
                        output_signature=tf.RaggedTensorSpec(shape=(INPUT_SIZE, 8, 1), dtype=tf.float32)).batch(batch_size=1).batch(batch_size=1),
                    use_multiprocessing=True                                                                                                                  
                )
                np.savetxt(self.data_path + f"/models/communities/community={community}/prediction.txt", self.hierarchy[city][community])
                pd.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), self.hierarchy[city][community])).to_csv(self.data_path+f"/models/communities/community={community}/evaluation.csv", index=False)
            self.hierarchy[city] = np.add.reduce(self.hierarchy[city].values())
            np.savetxt(self.data_path + f"/models/communities/city={community}/prediction.txt", self.hierarchy[city])
            pd.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), self.hierarchy[city])).to_csv(self.data_path+f"/models/communities/city={city}/evaluation.csv", index=False)
                    

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
        for city in self.hierarchy:
            self.hierarchy[city] = self.hierarchy[city].predict(
                x=tf.data.Dataset.from_generator(
                    generator=get_city_ec_input,
                    args=(self.data_path, city, test_windows),
                    output_signature=tf.RaggedTensorSpec(shape=(INPUT_SIZE, 8, 1), dtype=tf.float32)).batch(batch_size=1).batch(batch_size=1),
                use_multiprocessing=True                                                                                                                  
            )
            np.savetxt(self.data_path + f"/models/cities/city={city}/prediction.txt", self.hierarchy[city])
            pd.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), self.hierarchy[city])).to_csv(self.data_path+f"/models/cities/city={city}/evaluation.csv", index=False)
