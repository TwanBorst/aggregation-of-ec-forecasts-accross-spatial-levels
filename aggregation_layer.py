import os
from multiprocessing import Manager, Process, Semaphore
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import losses, metrics, models

from constants import *
from data_utils import (get_appliance_ec_input, get_appliance_ec_output,
                        get_city_ec_input, get_city_ec_output,
                        get_community_ec_input, get_community_ec_output,
                        get_household_ec_input, get_household_ec_output)


class ApplianceAggregationLayer:
    def __init__(self, hierarchy : dict, model_dir_name : str) -> None:
        self.hierarchy = hierarchy
        self.data_path = SAVE_DIR
        self.model_dir_name = model_dir_name
        
    def add_model(self, city, community, household, appliance, model):
        self.hierarchy[city][community][household][appliance] = model
        
    def from_disk(self):
        for city in self.hierarchy:
            for community in self.hierarchy[city]:
                for household in self.hierarchy[city][community]:
                    for appliance in self.hierarchy[city][community][household]:
                        self.hierarchy[city][community][household][appliance] = models.load_model(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/appliance={appliance}/model")
                        
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
                            np.savetxt(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/appliance={appliance}/prediction.txt", appliance_prediction)
                            pd.DataFrame.from_dict(compute_metrics(list(get_appliance_ec_output(self.data_path, household, appliance, test_windows)), appliance_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/appliance={appliance}/evaluation.csv", index=False)
                            if appliance in ["solar", "solar2", "battery1"]:
                                appliance_prediction = np.negative(appliance_prediction)
                            appliance_predictions.append(appliance_prediction)
                        household_prediction = np.add.reduce(appliance_predictions)
                        os.makedirs(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/", exist_ok=True)
                        np.savetxt(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/prediction.txt", household_prediction)
                        pd.DataFrame.from_dict(compute_metrics(list(get_household_ec_output(self.data_path, household, test_windows)), household_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/evaluation.csv", index=False)
                        household_predictions.append(household_prediction)
                    community_prediction = np.add.reduce(household_predictions)
                    os.makedirs(f"{self.data_path}/{self.model_dir_name}/appliances/community={community}/", exist_ok=True)
                    np.savetxt(f"{self.data_path}/{self.model_dir_name}/appliances/community={community}/prediction.txt", community_prediction)
                    pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/appliances/community={community}/evaluation.csv", index=False)
                    community_predictions.append(community_prediction)
                city_prediction = np.add.reduce(community_predictions)
                os.makedirs(f"{self.data_path}/{self.model_dir_name}/appliances/city={city}/", exist_ok=True)
                np.savetxt(f"{self.data_path}/{self.model_dir_name}/appliances/city={city}/prediction.txt", city_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/appliances/city={city}/evaluation.csv", index=False)

    def from_predictions(self, test_windows):
        for city, communities in self.hierarchy.items():
            community_predictions = []
            for community, households in communities.items():
                household_predictions = []
                for household, appliances in households.items():
                    appliance_predictions = []
                    for appliance, appliance_prediction in appliances.items():
                        appliance_prediction = np.loadtxt(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/appliance={appliance}/prediction.txt")
                        pd.DataFrame.from_dict(compute_metrics(list(get_appliance_ec_output(self.data_path, household, appliance, test_windows)), appliance_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/appliance={appliance}/evaluation.csv", index=False)
                        if appliance in ["solar", "solar2", "battery1"]:
                            appliance_prediction = np.negative(appliance_prediction)
                        appliance_predictions.append(appliance_prediction)
                    household_prediction = np.add.reduce(appliance_predictions)
                    os.makedirs(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/", exist_ok=True)
                    np.savetxt(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/prediction.txt", household_prediction)
                    pd.DataFrame.from_dict(compute_metrics(list(get_household_ec_output(self.data_path, household, test_windows)), household_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/appliances/household={household}/evaluation.csv", index=False)
                    household_predictions.append(household_prediction)
                community_prediction = np.add.reduce(household_predictions)
                os.makedirs(f"{self.data_path}/{self.model_dir_name}/appliances/community={community}/", exist_ok=True)
                np.savetxt(f"{self.data_path}/{self.model_dir_name}/appliances/community={community}/prediction.txt", community_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/appliances/community={community}/evaluation.csv", index=False)
                community_predictions.append(community_prediction)
            city_prediction = np.add.reduce(community_predictions)
            os.makedirs(f"{self.data_path}/{self.model_dir_name}/appliances/city={city}/", exist_ok=True)
            np.savetxt(f"{self.data_path}/{self.model_dir_name}/appliances/city={city}/prediction.txt", city_prediction)
            pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/appliances/city={city}/evaluation.csv", index=False)


class HouseholdAggregationLayer:
    def __init__(self, hierarchy : dict, model_dir_name : str) -> None:
        self.hierarchy = hierarchy
        self.data_path = SAVE_DIR
        self.model_dir_name = model_dir_name
        
    def add_model(self, city, community, household, model):
        self.hierarchy[city][community][household] = model
        
    def from_disk(self):
        for city in self.hierarchy:
            for community in self.hierarchy[city]:
                for household in self.hierarchy[city][community]:
                    self.hierarchy[city][community][household] = models.load_model(f"{self.data_path}/{self.model_dir_name}models/households/household={household}/model")
                        
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
                        np.savetxt(f"{self.data_path}/{self.model_dir_name}/households/household={household}/prediction.txt", household_prediction)
                        pd.DataFrame.from_dict(compute_metrics(list(get_household_ec_output(self.data_path, household, test_windows)), household_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/households/household={household}/evaluation.csv", index=False)
                        household_predictions.append(household_prediction)
                    community_prediction = np.add.reduce(household_predictions)
                    os.makedirs(f"{self.data_path}/{self.model_dir_name}/households/community={community}/", exist_ok=True)
                    np.savetxt(f"{self.data_path}/{self.model_dir_name}/households/community={community}/prediction.txt", community_prediction)
                    pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/households/community={community}/evaluation.csv", index=False)
                    community_predictions.append(community_prediction)
                city_prediction = np.add.reduce(community_predictions)
                os.makedirs(f"{self.data_path}/{self.model_dir_name}/households/city={city}/", exist_ok=True)
                np.savetxt(f"{self.data_path}/{self.model_dir_name}/households/city={city}/prediction.txt", city_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/households/city={city}/evaluation.csv", index=False)
    
    def from_predictions(self, test_windows : List[int]):
        for city, communities in self.hierarchy.items():
            community_predictions = []
            for community, households in communities.items():
                household_predictions = []
                for household, household_prediction in households.items():
                    household_prediction = np.loadtxt(f"{self.data_path}/{self.model_dir_name}/households/household={household}/prediction.txt")
                    pd.DataFrame.from_dict(compute_metrics(list(get_household_ec_output(self.data_path, household, test_windows)), household_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/households/household={household}/evaluation.csv", index=False)
                    household_predictions.append(household_prediction)
                community_prediction = np.add.reduce(household_predictions)
                os.makedirs(f"{self.data_path}/{self.model_dir_name}/households/community={community}/", exist_ok=True)
                np.savetxt(f"{self.data_path}/{self.model_dir_name}/households/community={community}/prediction.txt", community_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/households/community={community}/evaluation.csv", index=False)
                community_predictions.append(community_prediction)
            city_prediction = np.add.reduce(community_predictions)
            os.makedirs(f"{self.data_path}/{self.model_dir_name}/households/city={city}/", exist_ok=True)
            np.savetxt(f"{self.data_path}/{self.model_dir_name}/households/city={city}/prediction.txt", city_prediction)
            pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/households/city={city}/evaluation.csv", index=False)


class CommunityAggregationLayer:
    def __init__(self, hierarchy : dict, model_dir_name : str) -> None:
        self.hierarchy = hierarchy
        self.data_path = SAVE_DIR
        self.model_dir_name = model_dir_name
        
    def add_model(self, city, community, model):
        self.hierarchy[city][community] = model
        
    def from_disk(self):
        for city in self.hierarchy:
            for community in self.hierarchy[city]:
                self.hierarchy[city][community] = models.load_model(f"{self.data_path}/{self.model_dir_name}models/communities/community={community}/model")
                        
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
                    np.savetxt(f"{self.data_path}/{self.model_dir_name}/communities/community={community}/prediction.txt", community_prediction)
                    pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/communities/community={community}/evaluation.csv", index=False)
                    community_predictions.append(community_prediction)
                city_prediction = np.add.reduce(community_predictions)
                os.makedirs(f"{self.data_path}/{self.model_dir_name}/communities/city={city}/", exist_ok=True)
                np.savetxt(f"{self.data_path}/{self.model_dir_name}/communities/city={city}/prediction.txt", city_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/communities/city={city}/evaluation.csv", index=False)
    
    def from_predictions(self, test_windows):
        for city, communities in self.hierarchy.items():
            community_predictions = []
            for community, community_prediction in communities.items():
                community_prediction = np.loadtxt(f"{self.data_path}/{self.model_dir_name}/communities/community={community}/prediction.txt")
                pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/communities/community={community}/evaluation.csv", index=False)
                community_predictions.append(community_prediction)
            city_prediction = np.add.reduce(community_predictions)
            os.makedirs(f"{self.data_path}/{self.model_dir_name}/communities/city={city}/", exist_ok=True)
            np.savetxt(f"{self.data_path}/{self.model_dir_name}/communities/city={city}/prediction.txt", city_prediction)
            pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/communities/city={city}/evaluation.csv", index=False)
                   

class CityAggregationLayer:
    def __init__(self, hierarchy : dict, model_dir_name : str) -> None:
        self.hierarchy = hierarchy
        self.data_path = SAVE_DIR
        self.model_dir_name = model_dir_name
        
    def add_model(self, city, model):
        self.hierarchy[city] = model
        
    def from_disk(self):
        for city in self.hierarchy:
            self.hierarchy[city] = models.load_model(f"{self.data_path}/{self.model_dir_name}/cities/city={city}/model")
                        
    def evaluate(self, test_windows : List[int]):
        with Manager() as manager:
            semaphore = Semaphore(MAX_PARALLEL_TRAINING_MODELS)
            processes = []
            result_dict = manager.dict()
            for city, model in self.hierarchy.items():
                process = Process(target=predict,
                                  args=(get_city_ec_input,
                                        (self.data_path, city, test_windows),
                                        model, result_dict, city, semaphore))
                process.start()
                processes.append(process)
            self.hierarchy = result_dict
            for process in processes:
                process.join()
            for city, city_prediction in self.hierarchy.items():
                np.savetxt(f"{self.data_path}/{self.model_dir_name}/cities/city={city}/prediction.txt", city_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/cities/city={city}/evaluation.csv", index=False)
    
    def from_predictions(self, test_windows):
        for city, city_prediction in self.hierarchy.items():
            city_prediction = np.loadtxt(f"{self.data_path}/{self.model_dir_name}/cities/city={city}/prediction.txt")
            pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(f"{self.data_path}/{self.model_dir_name}/cities/city={city}/evaluation.csv", index=False)

def predict(generator, generator_args, model, result_dict, key, semaphore):
    with semaphore:
        result_dict[key] = model.predict(x=tf.data.Dataset.from_generator(generator=generator,
                                                                          args=generator_args,
                                                                          output_signature=tf.TensorSpec(shape=(INPUT_SIZE, 8),
                                                                                                         dtype=tf.float32))
                                                          .batch(batch_size=BATCH_SIZE),
                                         use_multiprocessing=True)
        
def compute_metrics(y_true, y_pred):
    rmse = metrics.RootMeanSquaredError()
    rmse.update_state(np.array(y_true).flatten(), np.array(y_pred).flatten())
    return {'mae': losses.mae(y_true, y_pred), 'rmse': rmse.result().numpy()}    