from typing import List
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sensor_type import SENSOR_NAMES
from math import ceil, floor

def availability_converter(val : str):
    return 0 if val == '' else int(val.strip('%'))

def sensor_converter(val : str):
    return True if val == "yes" else False

metadata_dtype = {'dataid': int, 'active_record': bool, 'building_type': str, 'city': str, 'state': str, 'egauge_1s_min_time': object, 'egauge_1s_max_time': object, 'egauge_1min_min_time': object, 'egauge_1min_max_time': object}
data_dtype = {sensor : float for sensor in SENSOR_NAMES}
data_dtype['dataid'] = int
data_dtype['localminute'] = object

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

def generate_metadata(data_path: str, save_path: str, metadata_files: str = None, extra_metadata_cols : List[str] = ['city', 'state']):
    """
    Generates the correct metadata for all the 1-second csv data files found with globstring `data_path`. If the complete metadata is provided,
    the columns listed in `extra_metadata_cols` will be added to the generated metadata.

    Parameters:
    -----------
    data_path : str
        Globstring for data files.
    save_path : str
        Where to save the generated metadata.
    metadata_files : str, default None
        Globstring for complete metadata files.
    extra_metadata_cols : List[str], default ['city', 'state']
        Extra columns from complete metadata to be added to the generated metadata.
    """
    all_present = dd.Aggregation('all_present',
                                 chunk=lambda x: x.aggregate(lambda x: x.notna().any()),
                                 agg=lambda x: x.aggregate(lambda x: x.any()),
                                 finalize=lambda x: x.replace({True: "yes", False: ""}))

    collumns = ['dataid', 'localminute'] + SENSOR_NAMES
    agg_meta = {'egauge_1s_min_time': 'object', 'egauge_1s_max_time': 'object', 'egauge_1s_data_availability': 'int'}
    agg_meta.update({sensor: 'str' for sensor in SENSOR_NAMES})
    
    ddf : dd.DataFrame = dd.read_csv(data_path, dtype = data_dtype, blocksize=10e7, usecols=collumns)
    
    reorder_collumns = list(ddf.columns)
    reorder_collumns.insert(1, 'aggregation_abs_error')
    ddf = ddf.assign(aggregation_abs_error=(ddf[[x for x in SENSOR_NAMES if x not in ["grid", "solar", "solar2"]]].sum(axis=1) - ddf[["solar", "solar2", "grid"]].sum(axis=1)).abs().where(cond=ddf["grid"].notnull(), other=np.nan))[reorder_collumns]

    agg_dict = {'localminute': {'egauge_1s_min_time': 'min', 'egauge_1s_max_time': 'max', 'egauge_1s_data_availability': 'count'}, 'aggregation_abs_error': {'aggregation_abs_error_max' : 'max', 'aggregation_abs_error_mean': 'mean'}}
    agg_dict.update({sensor: {sensor: all_present} for sensor in SENSOR_NAMES})
    
    ddf = ddf.groupby('dataid', sort=False).agg(agg_dict)
    ddf.columns = ddf.columns.get_level_values(1)
    
    ddf['egauge_1s_min_time'] = dd.to_datetime(ddf['egauge_1s_min_time'], utc=True)
    ddf['egauge_1s_max_time'] = dd.to_datetime(ddf['egauge_1s_max_time'], utc=True)
    ddf['egauge_1s_data_availability'] = (ddf['egauge_1s_data_availability'] / ((ddf['egauge_1s_max_time'] - ddf['egauge_1s_min_time'] + np.timedelta64(1, "s")) / np.timedelta64(1, "s"))).apply(lambda x: str(floor(x*10000)/100)+"%", meta=('egauge_1s_data_availability', str))
    ddf['aggregation_abs_error_max'] = ddf['aggregation_abs_error_max'].apply(lambda x: str(floor(x*100)/100), meta=('aggregation_abs_error_max', str))
    ddf['aggregation_abs_error_mean'] = ddf['aggregation_abs_error_mean'].apply(lambda x: str(floor(x*100)/100), meta=('aggregation_abs_error_mean', str))
    
    merged = ddf
    if metadata_files is not None:
        merged_columns = extra_metadata_cols + list(ddf.columns)[0:]

        original_metadata : dd.Dataframe = dd.read_csv(metadata_files, skiprows=[1], dtype = metadata_dtype, blocksize=10e7, usecols=['dataid']+extra_metadata_cols)
        original_metadata = original_metadata.set_index('dataid')
        
        merged = ddf.merge(original_metadata, how='left', left_index=True, right_index=True)[merged_columns]
    
    merged.to_csv(save_path, single_file = True)