import os
from multiprocessing import Manager, Process, Semaphore
from typing import List

import numpy as np
import pandas as pd
from keras import models

from constants import *
from data_utils import get_appliance_ec_input, get_appliance_ec_output, get_city_ec_input, get_city_ec_output, get_community_ec_input, get_community_ec_output, get_household_ec_input, get_household_ec_output
from models import compute_metrics, predict


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
                        os.makedirs(self.data_path + f"/models/appliances/household={household}/", exist_ok=True)
                        np.savetxt(self.data_path + f"/models/appliances/household={household}/prediction.txt", household_prediction)
                        pd.DataFrame.from_dict(compute_metrics(list(get_household_ec_output(self.data_path, household, test_windows)), household_prediction)).to_csv(self.data_path+f"/models/appliances/household={household}/evaluation.csv", index=False)
                        household_predictions.append(household_prediction)
                    community_prediction = np.add.reduce(household_predictions)
                    os.makedirs(self.data_path + self.data_path + f"/models/appliances/community={city}/", exist_ok=True)
                    np.savetxt(self.data_path + f"/models/appliances/community={community}/prediction.txt", community_prediction)
                    pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(self.data_path+f"/models/appliances/community={community}/evaluation.csv", index=False)
                    community_predictions.append(community_prediction)
                city_prediction = np.add.reduce(community_predictions)
                os.makedirs(self.data_path + self.data_path + f"/models/appliances/city={city}/", exist_ok=True)
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
                    os.makedirs(self.data_path + f"/models/households/community={community}/", exist_ok=True)
                    np.savetxt(self.data_path + f"/models/households/community={community}/prediction.txt", community_prediction)
                    pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_output(self.data_path, community, test_windows)), community_prediction)).to_csv(self.data_path+f"/models/households/community={community}/evaluation.csv", index=False)
                    community_predictions.append(community_prediction)
                city_prediction = np.add.reduce(community_predictions)
                os.makedirs(self.data_path + f"/models/households/city={city}/", exist_ok=True)
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
                    pd.DataFrame.from_dict(compute_metrics(list(get_community_ec_input(self.data_path, community, test_windows)), community_prediction)).to_csv(self.data_path+f"/models/communities/community={community}/evaluation.csv", index=False)
                    community_predictions.append(community_prediction)
                city_prediction = np.add.reduce(community_predictions)
                os.makedirs(self.data_path + f"/models/communities/city={city}/", exist_ok=True)
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
                                        (self.data_path, city, test_windows),
                                        model, result_dict, city, semaphore))
                process.start()
                processes.append(process)
            self.hierarchy = result_dict
            for process in processes:
                process.join()
            for city, city_prediction in self.hierarchy.items():
                np.savetxt(self.data_path + f"/models/cities/city={city}/prediction.txt", city_prediction)
                pd.DataFrame.from_dict(compute_metrics(list(get_city_ec_output(self.data_path, city, test_windows)), city_prediction)).to_csv(self.data_path+f"/models/cities/city={city}/evaluation.csv", index=False)
