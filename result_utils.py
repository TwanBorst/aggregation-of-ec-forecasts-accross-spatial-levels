from multiprocessing import Pool
import dask.dataframe as dd
from constants import *
import re
import glob
import pandas as pd

class result_utils_class:
        
    def __init__(self, time_column_name : str = 'localminute', time_frequency : str = 'S', metadata_time_prefix : str = 'egauge_1s_') -> None:
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
        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/appliances/household*/appliance*/history_log_full.csv",
                                          dtype = {"epoch": int, "loss": float, "root_mean_squared_error": float},
                                          blocksize=10e7, include_path_column=True)
        data = data.rename(columns={"loss":"mae", "root_mean_squared_error": "rmse"}).map_partitions(func=self._partition_value_from_path, key="appliance")
        del data["path"]
        aggregated = data.groupby(by=["appliance", "epoch"]).aggregate('mean').reset_index().repartition("100MB")
        aggregated.to_csv(SAVE_DIR + "/appliances/training_metrics_by_appliance.csv", single_file=True, index=False)
        del aggregated["appliance"]
        aggregated.groupby(by=["epoch"]).aggregate('mean').reset_index().repartition("100MB").to_csv(SAVE_DIR + "/appliances/training_metrics.csv", single_file=True, index=False)
        
    def _average_metrics_per_epoch(self, data_path: str, save_path: str, key: str):
        data : dd.DataFrame = dd.read_csv(SAVE_DIR + data_path,
                                          dtype = {"epoch": int, "loss": float, "root_mean_squared_error": float},
                                          blocksize=10e7, include_path_column=True)
        data = data.rename(columns={"loss":"mae", "root_mean_squared_error": "rmse"})
        del data["path"]
        aggregated = data.groupby(by=["epoch"]).aggregate('mean').reset_index().repartition("100MB")
        aggregated.to_csv(SAVE_DIR + save_path + "/training_metrics.csv", single_file=True, index=False)            

    
    def _partition_value_from_path(self, df: dd.DataFrame, key: str):
        appliance_regex = fr"(?<=({key}=))(\w+)"
        df[key] = re.search(appliance_regex, df.head(1)["path"].iloc[0]).group()
        return df

    
    def summarize_evaluations(self):
        evaluations = glob.glob(SAVE_DIR + "/**/evaluation.csv", recursive=True)
        with Pool(8) as pool:
            pool.map(_summarize_evaluation, evaluations)
        self._average_appliance_evaluations()
        self._average_household_evaluations()
        self._average_community_evaluations()
        self._average_city_evaluations()

    def _average_appliance_evaluations(self):
        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/appliances/household=*/appliance=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        appliance_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()
        
        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/appliances/household=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        household_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/appliances/community=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        community_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/appliances/city=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        city_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()

        pd.DataFrame([appliance_metrics, household_metrics, community_metrics, city_metrics], index=["appliance", "household", "community", "city"], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(SAVE_DIR + "/appliance_evaluation.csv", index=True)

    def _average_household_evaluations(self):
        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/households/household=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        household_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/households/community=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        community_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/households/city=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        city_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()

        pd.DataFrame([household_metrics, community_metrics, city_metrics], index=["household", "community", "city"], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(SAVE_DIR + "/household_evaluation.csv", index=True)

    def _average_community_evaluations(self):
        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/communities/community=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        community_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()

        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/communities/city=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        city_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()

        pd.DataFrame([community_metrics, city_metrics], index=["community", "city"], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(SAVE_DIR + "/community_evaluation.csv", index=True)

    def _average_city_evaluations(self):
        data : dd.DataFrame = dd.read_csv(SAVE_DIR + "/cities/city=*/evaluation.csv", dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
        city_metrics = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()

        pd.DataFrame([city_metrics], index=["city"], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(SAVE_DIR + "/city_evaluation.csv", index=True)

def _summarize_evaluation(path):
    data : dd.DataFrame = dd.read_csv(path, dtype = {"mae": float, "rmse": float}, blocksize=10e7, usecols=["mae", "rmse"])
    lst = data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).mean(axis=0).compute().tolist() + data.rename(columns={"mae": "mae-mean", "rmse": "rmse-mean"}).std(axis=0).compute().tolist()
    pd.DataFrame([lst], columns=['mae-mean', 'rmse-mean', 'mae-std', 'rmse-std']).to_csv(path.replace("evaluation.csv", "evaluation_summarized.csv"), index=False)
