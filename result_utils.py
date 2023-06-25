import glob
import re
from multiprocessing import Pool

import dask.dataframe as dd
import pandas as pd

from constants import *


class result_utils_class:
        
    def __init__(self, time_column_name : str = 'localminute', time_frequency : str = 'S', metadata_time_prefix : str = 'egauge_1s_', model_dir_name = "models") -> None:
        r"""
        Data utils class.

        Parameters
        ----------
            time_column_name : `str`, default `"localminute"`)
                Datetime column name in data files. Valid values can be `"localminute"` or `"local_15min"`.
            time_frequency : `str`, default `'S'`
                Time frequency the data should have.
            metadata_time_prefix : `str`, default `"egauge_1s_"`
                Metadata column prefix name. Can be `"egauge_1s_"` or `"egauge_1min_"`. 
        """
        
        self.time_column_name = time_column_name
        self.time_frequency = time_frequency
        self.metadata_time_prefix = metadata_time_prefix
        self.model_dir_name = model_dir_name
        
        self.sensors_excluding_grid = [sensor for sensor in SENSOR_NAMES if sensor != "grid"]

        self.metadata_dtype = {'dataid': int, 'active_record': bool, 'building_type': str, 'city': str, 'state': str, metadata_time_prefix+'min_time': object, metadata_time_prefix + 'max_time': object}
        self.data_dtype = {sensor : float for sensor in SENSOR_NAMES}
        self.data_dtype['dataid'] = int
        self.data_dtype[time_column_name] = object
        
        availability_converter = lambda val: 0 if val == '' else int(val.strip('%'))

        self.metadata_converters = {metadata_time_prefix+'data_availability': availability_converter, metadata_time_prefix + 'data_availability': availability_converter}
        self.metadata_converters.update({sensor: (lambda val: True if val == "yes" else False) for sensor in SENSOR_NAMES})

        self.hierarchy = None

    def average_metrics_per_epoch(self):
        self._average_appliance_metrics_per_epoch()
        self._average_metrics_per_epoch("/households/household=*/history_log_full.csv", "/households/", "household")
        self._average_metrics_per_epoch("/communities/community=*/history_log_full.csv", "/communities/", "community")
        self._average_metrics_per_epoch("/cities/city=*/history_log_full.csv", "/cities/", "city")
    
    def _average_appliance_metrics_per_epoch(self):
        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/appliances/household*/appliance*/history_log_full.csv",
                                          dtype = {"epoch": int, "loss": float, "root_mean_squared_error": float},
                                          blocksize=10e7, include_path_column=True)
        data = data.rename(columns={"loss":"mae", "root_mean_squared_error": "rmse"}).map_partitions(func=self._partition_value_from_path, key="appliance")
        del data["path"]
        aggregated = data.groupby(by=["appliance", "epoch"]).aggregate('mean').reset_index().repartition("100MB")
        aggregated.to_csv(f"{SAVE_DIR}/{self.model_dir_name}/appliances/training_metrics_by_appliance.csv", single_file=True, index=False)
        del aggregated["appliance"]
        aggregated.groupby(by=["epoch"]).aggregate('mean').reset_index().repartition("100MB").to_csv(f"{SAVE_DIR}/{self.model_dir_name}/appliances/training_metrics.csv", single_file=True, index=False)
        
    def _average_metrics_per_epoch(self, data_path: str, save_path: str, key: str):
        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/{data_path}",
                                          dtype = {"epoch": int, "loss": float, "root_mean_squared_error": float},
                                          blocksize=10e7, include_path_column=True)
        data = data.rename(columns={"loss":"mae", "root_mean_squared_error": "rmse"})
        del data["path"]
        aggregated = data.groupby(by=["epoch"]).aggregate('mean').reset_index().repartition("100MB")
        aggregated.to_csv(f"{SAVE_DIR}/{self.model_dir_name}/{save_path}/training_metrics.csv", single_file=True, index=False)            

    
    def _partition_value_from_path(self, df: dd.DataFrame, key: str):
        appliance_regex = fr"(?<=({key}=))(\w+)"
        df[key] = re.search(appliance_regex, df.head(1)["path"].iloc[0]).group()
        return df

    
    def summarize_evaluations(self):
        evaluations = glob.glob(f"{SAVE_DIR}/{self.model_dir_name}/**/evaluation.csv", recursive=True)
        with Pool(8) as pool:
            pool.map(_summarize_evaluation, evaluations)

    def _average_appliance_evaluations(self):
        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/appliances/household=*/appliance=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        appliance_metrics = data.mean(axis=0).compute().tolist()
        
        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/appliances/household=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        household_metrics = data.mean(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/appliances/community=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        community_metrics = data.mean(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/appliances/city=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        city_metrics = data.mean(axis=0).compute().tolist()

        pd.DataFrame([appliance_metrics, household_metrics, community_metrics, city_metrics], index=["appliance", "household", "community", "city"], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(f"{SAVE_DIR}/{self.model_dir_name}/appliance_evaluation.csv", index=True)

    def _average_household_evaluations(self):
        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/households/household=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        household_metrics = data.mean(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/households/community=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        community_metrics = data.mean(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/households/city=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        city_metrics = data.mean(axis=0).compute().tolist()

        pd.DataFrame([household_metrics, community_metrics, city_metrics], index=["household", "community", "city"], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(f"{SAVE_DIR}/{self.model_dir_name}/household_evaluation.csv", index=True)

    def _average_community_evaluations(self):
        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/communities/community=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        community_metrics = data.mean(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/communities/city=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        city_metrics = data.mean(axis=0).compute().tolist()

        pd.DataFrame([community_metrics, city_metrics], index=["community", "city"], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(f"{SAVE_DIR}/{self.model_dir_name}/community_evaluation.csv", index=True)

    def _average_city_evaluations(self):
        data : dd.DataFrame = dd.read_csv(f"{SAVE_DIR}/{self.model_dir_name}/cities/city=*/evaluation_summarized.csv", blocksize=10e7, usecols=["mae-mean", "rmse-mean", "mae-std", "rmse-std"])
        city_metrics = data.mean(axis=0).compute().tolist()

        pd.DataFrame([city_metrics], index=["city"], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(f"{SAVE_DIR}/{self.model_dir_name}/city_evaluation.csv", index=True)

def _summarize_evaluation(path):
    data : dd.DataFrame = dd.read_csv(path, dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
    lst = data.mean(axis=0).compute().tolist() + data.std(axis=0).compute().tolist()
    pd.DataFrame([lst], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(path.replace("evaluation.csv", "evaluation_summarized.csv"), index=False)
