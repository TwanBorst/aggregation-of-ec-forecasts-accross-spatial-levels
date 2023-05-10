from typing import List
import numpy as np
import pandas as pd
from dask.distributed import Client
import dask.dataframe as dd
from sensor_type import SensorType, SENSOR_NAMES

def availability_converter(val : str):
    return 0 if val == '' else int(val.strip('%'))

def sensor_converter(val : str):
    return True if val == "yes" else False

metadata_dtype = {'dataid': int, 'active_record': bool, 'building_type': str, 'city': str, 'state': str, 'egauge_1s_min_time': object, 'egauge_1s_max_time': object, 'egauge_1min_min_time': object, 'egauge_1min_max_time': object}

metadata_converters = {'egauge_1min_data_availability': availability_converter, 'egauge_1s_data_availability': availability_converter}
metadata_converters.update({sensor: sensor_converter for sensor in SENSOR_NAMES})

def process_data(files: str, save_path: str, sensors: List[str] = None, data_ids: List[int] = None, from_time: pd.Timestamp = None, end_time: pd.Timestamp = None):
    
    collumns = ['dataid', 'localminute'] + (sensors if sensors is not None else SENSOR_NAMES)
    data_dtype = {sensor : float for sensor in (sensors if sensors is not None else SENSOR_NAMES)}
    data_dtype['dataid'] = int
    data_dtype['localminute'] = object
    
    ddf = dd.read_csv(files, dtype = data_dtype, blocksize=10e7, usecols=collumns)
    if data_ids is not None:
        ddf = ddf[ddf['dataid'].isin(data_ids)]
    
    if from_time is not None or end_time is not None:
        ddf['localminute'].map_partitions(pd.to_datetime, utc=True, meta={'localminute': 'datetime64[s]'})
        if from_time is not None:
            ddf = ddf[ddf['localminute'] >= str(from_time)]
        if end_time is not None:
            ddf = ddf[ddf['localminute'] <= str(end_time)]
    
    ddf.to_csv(save_path)

def get_data_ids_from_metadata(data_path: str = '../data/*/metadata.csv', required_sensor_data : List[str] = [],
                     min_1s_completeness: int = 0, min_1min_completeness: int = 0,
                     min_1s_time: pd.Timestamp = None, max_1s_time: pd.Timestamp = None,
                     min_1min_time: pd.Timestamp = None, max_1min_time: pd.Timestamp = None) -> pd.DataFrame :
    r"""
    Returns a list of all data-ids (home-resident pairs) matching the given filters.
    
    Parameters:
    -----------
    data_path : str, default '../data/*/metadata.csv'
        Globstring for metadata files
    required_sensor_data : List[str], default []
        List of sensors that have to be marked as 'yes'
    min_1s_completeness : pd.Timestamp, default None
        Minimal 1-second data completeness. Value between 0 and 100.
    min_1min_completeness : pd.Timestamp, default None
        Minimal 1-minute data completeness. Value between 0 and 100.
    min_1s_time : pd.Timestamp, default None
        Timestamp after which 1-second data needs to be present
    max_1s_time : pd.Timestamp, default None
        Timestamp up to which 1-second data needs to be present
    min_1s_time : pd.Timestamp, default None
        Timestamp after which 1-minute data needs to be present
    max_1s_time : pd.Timestamp, default None
        Timestamp up to which 1-minute data needs to be present
        
    Returns:
    --------
    List of data-ids (home-resident pairs) matching the given filters.
    """
    collumns = ["dataid", "egauge_1s_data_availability", "egauge_1min_data_availability", "egauge_1s_min_time", "egauge_1s_max_time", "egauge_1min_min_time", "egauge_1min_max_time"] + SENSOR_NAMES
    query : List[str] = []
    for sensor in required_sensor_data:
        query.append(f'{sensor} == True')
    if min_1s_completeness > 0.0:
        query.append(f'egauge_1s_data_availability >= {min_1s_completeness}')
    if min_1min_completeness > 0.0:
        query.append(f'egauge_1min_data_availability >= {min_1min_completeness}')
    
    ddf = dd.read_csv(data_path, dtype = metadata_dtype, skiprows=[1], converters = metadata_converters, blocksize=25e8, usecols=collumns)        
    
    if len(query) > 0:
        ddf = ddf.query(" and ".join(query))

    if min_1s_time is not None:
        ddf['egauge_1s_min_time'].map_partitions(pd.to_datetime, utc=True, meta={'egauge_1s_min_time': 'datetime64[s]'})
        ddf = ddf[ddf['egauge_1s_min_time'] <= str(min_1s_time)]
    if min_1min_time is not None:
        ddf['egauge_1min_min_time'].map_partitions(pd.to_datetime, utc=True, meta={'egauge_1min_min_time': 'datetime64[s]'})
        ddf = ddf[ddf['egauge_1min_min_time'] <= str(min_1min_time)]
    if max_1s_time is not None:
        ddf['egauge_1s_max_time'].map_partitions(pd.to_datetime, utc=True, meta={'egauge_1s_max_time': 'datetime64[s]'})
        ddf = ddf[ddf['egauge_1s_max_time'] >= str(max_1s_time)]
    if max_1min_time is not None:
        ddf['egauge_1s_max_time'].map_partitions(pd.to_datetime, utc=True, meta={'egauge_1min_max_time': 'datetime64[s]'})
        ddf = ddf[ddf['egauge_1min_max_time'] >= str(max_1min_time)]
    
    return ddf['dataid'].compute().to_list()
