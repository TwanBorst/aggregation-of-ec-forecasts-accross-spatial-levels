from typing import List
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sensor_type import SENSOR_NAMES
from math import floor
import pyarrow as pa
import tempfile

pd.options.display.max_rows = 100

class data_utils_class:
        
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

        self.metadata_converters = {'egauge_1min_data_availability': availability_converter, metadata_time_prefix + 'data_availability': availability_converter}
        self.metadata_converters.update({sensor: (lambda val: True if val == "yes" else False) for sensor in SENSOR_NAMES})


    def process_data(self, files: str, metadata_files:str, save_path: str, from_time: pd.Timestamp, end_time: pd.Timestamp, sensors: List[str] = None):
        r"""
        Used to process the original unpacked data. Missing rows (seconds) are filled with the last value of the 5 seconds that came before.
        If there is no pre-existing value within those 5 seconds, the value will be set to 0.
        Additionally values that represent over 1MW are removed and treated as if they were empty from the start.
        

        Paramters
        ---------
        files : `str`
            The globstring ("path/to/files/*.csv") to the orignal unpacked csv files
        metadata_files : `str`
            The path to the metadata file made by `generate_metadata`
        save_path : `str`
            The first location that can be used for storing intermediate results
        from_time : `pd.Timestamp`
            Time after which you want the 1-second data to start
        end_time : `pd.Timestamp`
            Time after which you want the 1-second data to end
        sensors : `List[str]`, default `None`
            List of sensors that will be in the final result. If the value is `None` all sensors are included.
        """
        sensors_excluding_grid = sensors if sensors is not None else self.sensors_excluding_grid
        collumns = ['dataid', self.time_column_name] + sensors_excluding_grid
        data_dtype = {sensor : float for sensor in sensors_excluding_grid}
        data_dtype['dataid'] = int
        data_dtype[self.time_column_name] = object
        
        ddf : dd.DataFrame = dd.read_csv(files, dtype = data_dtype, blocksize=10e7, usecols=collumns)
        original_metadata : dd.DataFrame = dd.read_csv(metadata_files, dtype = self.metadata_dtype, blocksize=10e7, usecols=['dataid', 'state', self.metadata_time_prefix+'min_time', self.metadata_time_prefix+'max_time'])
        original_metadata[self.metadata_time_prefix+'max_time'] = dd.to_datetime(original_metadata[self.metadata_time_prefix+'max_time'], utc=True)
        original_metadata[self.metadata_time_prefix+'min_time'] = dd.to_datetime(original_metadata[self.metadata_time_prefix+'min_time'], utc=True)

        
        # Drop dataids that don't have data within the from and end dates while ignoring years.
        original_metadata['delta_year'] = (end_time.year - original_metadata[self.metadata_time_prefix+'max_time'].dt.year).astype('timedelta64[Y]')
        original_metadata['delta_year'] = original_metadata['delta_year'].where(original_metadata[self.metadata_time_prefix+'max_time'] + original_metadata['delta_year'] >= end_time, original_metadata['delta_year'] + np.timedelta64(1, 'Y'))
        filter_and_offset = original_metadata[original_metadata[self.metadata_time_prefix+'min_time'] + original_metadata['delta_year'] <= from_time][['dataid', 'delta_year']]
        ddf = ddf.merge(filter_and_offset, how="inner", on=["dataid"])
        
        # Align the recorded datetimes of the data to the same year
        ddf[self.time_column_name] = dd.to_datetime(ddf[self.time_column_name], utc=True)
        ddf[self.time_column_name] = ddf[self.time_column_name] + ddf['delta_year']
        del ddf['delta_year']

        dd.to_parquet(ddf.set_index('dataid').repartition(partition_size="100MB").reset_index(), save_path+"/temp/time-adjusted", write_index=False, partition_on=["dataid"], name_function=lambda x: f"data-{x}.parquet", schema={self.time_column_name: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()}, overwrite=True)
        dataids: List[int] = original_metadata['dataid'].compute().values.tolist()
        del original_metadata
        
        date_range = dd.from_pandas(pd.date_range(start=from_time, end=end_time, freq=self.time_frequency).to_frame(name=self.time_column_name), sort=False, npartitions=1)
        dd.to_parquet(date_range.repartition(partition_size="100MB"), save_path+"/temp/date-range", write_index=False, name_function=lambda x: f"data-{x}.parquet", schema={self.time_column_name: pa.timestamp(unit='s', tz='UTC')}, overwrite=True)
        del date_range
        
        total = len(dataids)
        
        for progress, dataid in enumerate(dataids):
            print("\n-------------------------------", f"|     Starting with dataid={dataid}.  progress: {progress}/{total}    |", "--------------------------------\n")
            
            dataid_ddf = dd.read_parquet(save_path+"/temp/time-adjusted",
                                        filters=[('dataid', '==', dataid)],
                                        columns=sensors_excluding_grid+[self.time_column_name])
            date_range = dd.read_parquet(save_path+"/temp/date-range")
            merged = date_range.merge(dataid_ddf, how='left', on=[self.time_column_name]).assign(dataid=dataid)
            dd.to_parquet(merged.repartition(partition_size="100MB"), save_path+"/temp/rows-filled",
                        write_index=False, partition_on=["dataid"],
                        name_function=lambda x: f"data-{x}.parquet",
                        schema={self.time_column_name: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()},
                        append=True)
            
            print("\n--------------------------------", f"|      Done with dataid={dataid}       |", "--------------------------------\n")
                                            
                
        # If a sensor reports 1MW of charging we interperet it as an error and remove the datapoint
        # Fill in small gaps of 5 seconds with the previously encountered value and fill the rest up with 0's
        for progress, dataid in enumerate(dataids):
            print("\n-------------------------------", f"|     Starting with dataid={dataid}.  progress: {progress}/{total}    |", "--------------------------------\n")
            
            dataid_ddf = dd.read_parquet(save_path+"/temp/rows-filled",
                                        filters=[('dataid', '==', dataid)],
                                        columns=sensors_excluding_grid+["dataid", self.time_column_name]).set_index(self.time_column_name)
            dataid_ddf[sensors_excluding_grid] = dataid_ddf[sensors_excluding_grid].mask(dataid_ddf[sensors_excluding_grid] > 1000, np.nan).fillna(method="ffill", limit=5).fillna(0)
            
            dataid_ddf['dataid'] = dataid_ddf['dataid'].cat.as_ordered()
            dd.to_parquet(dataid_ddf.reset_index(), save_path+"/final_appliance",
                        write_index=False, partition_on=["dataid"],
                        name_function=lambda x: f"data-{x}.parquet",
                        schema={self.time_column_name: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()},
                        append=True)
            
            print("\n--------------------------------", f"|      Done with dataid={dataid}       |", "--------------------------------\n")


    def generate_metadata(self, data_path: str, save_path: str, metadata_files: str, extra_metadata_cols : List[str] = [], community_size : int = 5):
        """
        Generates the correct metadata for all the 1-second csv data files found with globstring `data_path`. If the complete metadata is provided,
        the columns listed in `extra_metadata_cols` will be added to the generated metadata.

        Parameters:
        -----------
        data_path : `str`
            Globstring for data files.
        save_path : `str`
            Where to save the generated metadata.
        metadata_files : `str`, default `None`
            Globstring for complete metadata files.
        extra_metadata_cols : `List[str]`, default `[]`
            Extra columns from complete metadata to be added to the generated metadata next to `['city', 'state']`.
        """
        all_present = dd.Aggregation('all_present',
                                    chunk=lambda x: x.aggregate(lambda x: x.notna().any()),
                                    agg=lambda x: x.aggregate(lambda x: x.any()),
                                    finalize=lambda x: x.replace({True: "yes", False: ""}))

        collumns = ['dataid', self.time_column_name] + SENSOR_NAMES
        
        ddf : dd.DataFrame = dd.read_csv(data_path, dtype = self.data_dtype, blocksize=10e7, usecols=collumns)
        
        reorder_collumns = list(ddf.columns)
        reorder_collumns.insert(1, 'aggregation_abs_error')
        ddf = ddf.assign(aggregation_abs_error=(ddf[[x for x in SENSOR_NAMES if x not in ["grid", "solar", "solar2", "battery1"]]].sum(axis=1) - ddf[["solar", "solar2", "battery1", "grid"]].sum(axis=1)).abs().where(cond=ddf["grid"].notnull(), other=np.nan))[reorder_collumns]

        agg_dict = {self.time_column_name: {self.metadata_time_prefix + 'min_time': 'min', self.metadata_time_prefix + 'max_time': 'max', self.metadata_time_prefix + 'data_availability': 'count'}, 'aggregation_abs_error': {'aggregation_abs_error_max' : 'max', 'aggregation_abs_error_mean': 'mean'}}
        agg_dict.update({sensor: {sensor: all_present} for sensor in SENSOR_NAMES})
        
        ddf = ddf.groupby('dataid', sort=False).agg(agg_dict)
        ddf.columns = ddf.columns.get_level_values(1)
        
        ddf[self.metadata_time_prefix + 'min_time'] = dd.to_datetime(ddf[self.metadata_time_prefix + 'min_time'], utc=True)
        ddf[self.metadata_time_prefix + 'max_time'] = dd.to_datetime(ddf[self.metadata_time_prefix + 'max_time'], utc=True)

        ddf[self.metadata_time_prefix + 'data_availability'] = (ddf[self.metadata_time_prefix + 'data_availability'] / ((ddf[self.metadata_time_prefix + 'max_time'] - ddf[self.metadata_time_prefix + 'min_time'] + pd.Timedelta(self.time_frequency)) / pd.Timedelta(self.time_frequency))).apply(lambda x: str(floor(x*10000)/100)+"%", meta=(self.metadata_time_prefix + 'data_availability', str))
        ddf['aggregation_abs_error_max'] = ddf['aggregation_abs_error_max'].apply(lambda x: str(floor(x*100)/100), meta=('aggregation_abs_error_max', str))
        ddf['aggregation_abs_error_mean'] = ddf['aggregation_abs_error_mean'].apply(lambda x: str(floor(x*100)/100), meta=('aggregation_abs_error_mean', str))
        
        merged_columns = ['city', 'state'] + extra_metadata_cols + list(ddf.columns)[0:]

        original_metadata : dd.DataFrame = dd.read_csv(metadata_files, skiprows=[1], dtype = self.metadata_dtype, blocksize=10e7, usecols=['dataid', 'city', 'state']+extra_metadata_cols)
        original_metadata = original_metadata.set_index('dataid')
        
        ddf : dd.DataFrame = ddf.merge(original_metadata, how='left', left_index=True, right_index=True)[merged_columns]
            
        ddf['community'] = (ddf.groupby('city').cumcount() % community_size == 0).replace({True: 1, False: 0})
        ddf = ddf.sort_values(['city', 'dataid'])
        ddf['community'] = ddf['community'].cumsum()

        ddf.to_csv(save_path, single_file = True)
        
        
    def generate_aggregated_data(self, data_path: str, metadata: str):
        r"""
        Aggregates the data to a houshold, community, and city level and stores it to the `data_path` folder.

        Parameters:
        ----------
            data_path : `str`
                The path to the root folder of the data.
            metadata : `str`
                The path to the generated metadata file.
        """
        ddf: dd.DataFrame = dd.read_parquet(data_path+"/final_appliance", columns=['dataid', self.time_column_name] + self.sensors_excluding_grid)

        ddf_household = ddf[['dataid', self.time_column_name]].assign(total=ddf[[x for x in SENSOR_NAMES if x not in ["solar", "solar2", "battery1", "grid"]]].sum(axis=1) - ddf[["solar", "solar2", "battery1"]].sum(axis=1))
        dd.to_parquet(ddf_household, data_path+"/final_household",
                        write_index=False, partition_on=["dataid"],
                        name_function=lambda x: f"data-{x}.parquet",
                        schema={self.time_column_name: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()})
        
        meta_ddf: dd.DataFrame = dd.read_csv(metadata, dtype = self.metadata_dtype, blocksize=10e7, usecols=['dataid', 'community', 'city'])
        ddf_household_merged = ddf_household.merge(meta_ddf, how='left', on=["dataid"])
        del ddf
        del meta_ddf
        del ddf_household
    
        total_community = ddf_household_merged.groupby(['community', self.time_column_name])['total'].sum()
        ddf_community = ddf_household_merged.drop(columns=['total', 'dataid', 'city']).drop_duplicates(subset=['community']).join(total_community, on=['community', self.time_column_name])
        dd.to_parquet(ddf_community.repartition(partition_size="100MB"), data_path+"/final_community",
                        write_index=False, partition_on=["community"],
                        name_function=lambda x: f"data-{x}.parquet",
                        schema={self.time_column_name: pa.timestamp(unit='s', tz='UTC'), 'community': pa.int32()})
        del total_community
        del ddf_community
        
        total_city = ddf_household_merged.groupby(['city', self.time_column_name])['total'].sum()
        ddf_city = ddf_household_merged.drop(columns=['total', 'dataid', 'community']).drop_duplicates(subset=['city']).join(total_city, on=['city', self.time_column_name])
        dd.to_parquet(ddf_city.repartition(partition_size="100MB"), data_path+"/final_city",
                        write_index=False, partition_on=["city"],
                        name_function=lambda x: f"data-{x}.parquet",
                        schema={self.time_column_name: pa.timestamp(unit='s', tz='UTC'), 'city': pa.string()})
        
        