import argparse
import os

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa

from dask.distributed import Client
from math import floor

from constants import *


def get_arguments():
    """
    The arguments passed when calling the method.

    Returns:
    """
    parser = argparse.ArgumentParser(description='sequence to point learning example for NILM')

    parser.add_argument('--data_dir', type=str, default=DATA_PATH, help='The directory containing the Pecan '
                                                                        'Street data')
    parser.add_argument('--house_number', type=int, default=1, help='The house to train for')
    parser.add_argument('--aggregate_mean', type=int, default=AGG_MEAN, help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std', type=int, default=AGG_STD, help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH, help='The directory to store the training data')
    return parser.parse_args()


# Save arguments as global variables
ARGS = get_arguments()
HOUSE = ARGS.house_number
print("The house to train for:", HOUSE)


def main():
    dask.config.set(temporary_directory=DASK_TMP_DIR)

    client = Client(n_workers=4, threads_per_worker=2, memory_limit="10GB")
    print(client)

    print("\n----------------------------", f"|        Metadata shit         |", "----------------------------\n")

    columns = ['dataid', TIME_COLUMN_NAME] + SENSOR_NAMES                   # All column names

    data = dd.read_csv(NEW_YORK_15MIN, usecols=columns)                     # The data

    reorder_columns = list(data.columns)                                    # List of columns
    reorder_columns.insert(1, 'aggregation_abs_error')                      # Insert column at position 1 (zero indexed)

    # Calculates the aggregation absolute error, and adds this column at position 1.
    data = data.assign(aggregation_abs_error=(
            data[[x for x in SENSOR_NAMES if x not in ["grid", "solar", "solar2", "battery1"]]].sum(axis=1) -
            data[["solar", "solar2", "battery1", "grid"]].sum(axis=1)).abs().where(cond=data["grid"].notnull(),
                                                                                   other=np.nan))
    data = data[reorder_columns]

    # Aggregation in terms of operations on Pandas dataframes in a map-reduce style. You need to specify what operation
    # to do on each chunk of data, how to combine those chunks of data together, and then how to finalize the result.
    # Name      'all present'
    # Chuck     Check if there is at least one not null value present in the column for that chunk.
    # Agg       Check if there is at least one not null value present in the column for all chunks.
    # Finalize  Put a string "yes" in all columns that have at least 1 value.
    all_present = dd.Aggregation('all_present',
                                 chunk=lambda x: x.aggregate(lambda y: y.notna().any()),
                                 agg=lambda x: x.aggregate(lambda y: y.any()),
                                 finalize=lambda x: x.replace({True: "yes", False: ""}))

    # Sets some methods used for aggregating the columns.
    agg_dict = {TIME_COLUMN_NAME: {METADATA_TIME_PREFIX + 'min_time': 'min',
                                   METADATA_TIME_PREFIX + 'max_time': 'max',
                                   METADATA_TIME_PREFIX + 'data_availability': 'count'},
                'aggregation_abs_error': {'aggregation_abs_error_max': 'max',
                                          'aggregation_abs_error_mean': 'mean'}
                }
    agg_dict.update({sensor: {sensor: all_present} for sensor in SENSOR_NAMES})

    # Groups everything by 'dataid'. For the columns time and aggregation it gives some values. For the other columns it
    # tells whether they have values. After the columns are updated with the added names.
    data = data.groupby('dataid', sort=False).agg(agg_dict)
    data.columns = data.columns.get_level_values(1)

    # Sets the time of max_time and min_time columns to a time format.
    data[METADATA_TIME_PREFIX + 'min_time'] = dd.to_datetime(data[METADATA_TIME_PREFIX + 'min_time'], utc=True)
    data[METADATA_TIME_PREFIX + 'max_time'] = dd.to_datetime(data[METADATA_TIME_PREFIX + 'max_time'], utc=True)

    # Sets the data availability as a percentage.
    data[METADATA_TIME_PREFIX + 'data_availability'] = (
            data[METADATA_TIME_PREFIX + 'data_availability'] /
            ((data[METADATA_TIME_PREFIX + 'max_time'] - data[METADATA_TIME_PREFIX + 'min_time'] + pd.Timedelta(
                TIME_FREQUENCY)) / pd.Timedelta(TIME_FREQUENCY))
    ).apply(lambda x: str(floor(x * 10000) / 100) + "%", meta=(METADATA_TIME_PREFIX + 'data_availability', str))

    # Rounds the absolute error to two decimal places.
    data['aggregation_abs_error_max'] = data['aggregation_abs_error_max'].apply(
        lambda x: str(floor(x * 100) / 100),
        meta=('aggregation_abs_error_max', str)
    )
    data['aggregation_abs_error_mean'] = data['aggregation_abs_error_mean'].apply(
        lambda x: str(floor(x * 100) / 100),
        meta=('aggregation_abs_error_mean', str)
    )

    # Reads the city information for all 'dataids' and adds this information into the data Dataframe.
    metadata = dd.read_csv(
        METADATA,
        skiprows=[1],
        dtype=METADATA_TYPE,
        blocksize=10e7,
        usecols=['dataid', 'city']
    )

    metadata = metadata.set_index('dataid')
    data = data.merge(metadata, how='left', left_index=True, right_index=True)

    # Re-orders the columns, such that the city if the first column
    columns = data.columns.tolist()
    data = data[columns[-1:] + columns[:-1]]

    # Drop dataids that don't have data within the from and end dates while ignoring years.
    # data['delta_year'] = (END_TIME.year - data[METADATA_TIME_PREFIX + 'max_time'].dt.year).astype('timedelta64[Y]')
    # data['delta_year'] = data['delta_year'].where(data[METADATA_TIME_PREFIX + 'max_time'] +
    #       data['delta_year'] >= END_TIME, data['delta_year'] + np.timedelta64(1, 'Y'))
    # data = data[data[METADATA_TIME_PREFIX + 'min_time'] + data['delta_year'] <= FROM_TIME]

    # Sort the data by city and dataid, then it creates a column 'community' where it adds a 1 to every house within the
    # community that leaves no remainder.
    data = data.sort_values(['city', 'dataid'])
    data['community'] = (data.groupby('city').cumcount() % COMMUNITY_SIZE == 0).replace({True: 1, False: 0})

    # Adds a number to every house indicating to what community it belongs.
    data['community'] = data['community'].cumsum()

    # Counts how often a community occurs, then it resets the index. The 'index' column tells you what community it is,
    # and the community column has the counts of the houses in the community.
    community_sizes = data['community'].value_counts().reset_index()

    # Filter that select only the community index where the community size equals the calculated community size.
    community_filter = community_sizes[community_sizes['count'] == COMMUNITY_SIZE]['community'].compute()
    data = data[data['community'].isin(community_filter)]

    data.to_csv(CUSTOM_METADATA, single_file=True)

    print("\n----------------------------", f"|     Processing the data      |", "----------------------------\n")

    sensors = [sensor for sensor in SENSOR_NAMES if sensor != "grid"]   # All sensors without grid
    columns = ['dataid', TIME_COLUMN_NAME] + sensors                    # All column names

    # data_dtype = {sensor: float for sensor in sensors}                  # Dictionary with column types
    # data_dtype['dataid'] = int                                          # Set type of 'dataid' to int
    # data_dtype[TIME_COLUMN_NAME] = object                               # Set type of time column to object

    # Load the data and metadata from file
    data = dd.read_csv(NEW_YORK_15MIN, blocksize=10e7, usecols=columns)
    metadata = dd.read_csv(CUSTOM_METADATA, dtype=METADATA_TYPE, blocksize=10e7, usecols=['dataid'])

    # Filter 'dataid' not in metadata
    data = data.merge(metadata, how="inner", on=["dataid"])

    # Set the type for the time column
    data[TIME_COLUMN_NAME] = dd.to_datetime(data[TIME_COLUMN_NAME], utc=True)

    dd.to_parquet(
        data.set_index('dataid').repartition(partition_size="100MB").reset_index(),
        SAVE_PATH + "temp/rows_filled/",
        write_index=False,
        partition_on=["dataid"],
        name_function=lambda x: f"data-{x}.parquet",
        schema={TIME_COLUMN_NAME: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()},
        overwrite=True
    )

    # Outputs a list of all different dataids and deletes the metadata Dataframe after.
    dataids = metadata['dataid'].compute().values.tolist()
    del metadata

    total = len(dataids)

    print("\n----------------------------", f"| Outlier detection and missing |", "----------------------------")
    print("----------------------------", f"|     value filling started     |", "----------------------------\n")

    # If a sensor reports 1MW of charging we interperet it as an error and remove the datapoint
    # Fill in small gaps of 5 seconds with the previously encountered value and fill the rest up with 0's
    for progress, dataid in enumerate(dataids):

        print("\n----------------------------", f"|  Starting with dataid = {dataid}  |",
              "----------------------------")
        print("----------------------------", f"|        progress: {progress}/{total}         |",
              "----------------------------\n")

        # Reads the data related to one dataid / house and sets the index to the time
        dataid_ddf = dd.read_parquet(
            SAVE_PATH + "temp/rows_filled/",
            filters=[('dataid', '==', dataid)],
            columns=sensors + ["dataid", TIME_COLUMN_NAME]
        ).set_index(TIME_COLUMN_NAME)

        mean = dataid_ddf[sensors].mean(axis=1)                         # The mean value of the house.
        std = dataid_ddf[sensors].std(axis=1)                           # The standard deviation of the house.

        # Calculates if the individual values within the dataframe contain outliers, and replaces them with NaN's. Then
        # it goes over the columns, and fills at most 5 subsequent values with the previous value. If there are more
        # than 5 missing values it fills it with 0.
        dataid_ddf[sensors] = dataid_ddf[sensors].mask(abs((dataid_ddf[sensors] - mean) / std) > 3, np.nan) \
            .fillna(method="ffill", limit=5).fillna(0)

        # Converts the 'dataid' column into an ordered categorical data type
        dataid_ddf['dataid'] = dataid_ddf['dataid'].cat.as_ordered()

        dd.to_parquet(
            dataid_ddf.reset_index(),
            SAVE_PATH + "temp/timestamp_extracted/",
            write_index=False,
            partition_on=["dataid"],
            name_function=lambda x: f"data-{x}.parquet",
            schema={TIME_COLUMN_NAME: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()},
            append=True
        )

        print("\n----------------------------", f"|    Done with dataid = {dataid}     |",
              "----------------------------\n")

    print("\n----------------------------", f"| Outlier detection and missing |", "----------------------------")
    print("----------------------------", f"|      value filling done       |", "----------------------------\n")

    print("\n----------------------------", f"| Repartitioning appliance data |", "----------------------------\n")

    # Repartition appliance data
    dataid_ddf = dd.read_parquet(SAVE_PATH + "temp/timestamp_extracted/")

    dd.to_parquet(
        dataid_ddf.repartition(partition_size="100MB"),
        SAVE_PATH + "final_appliance/",
        write_index=False,
        partition_on=["dataid"],
        name_function=lambda x: f"data-{x}.parquet",
        schema={TIME_COLUMN_NAME: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()}
    )

    print("\n----------------------------", f"| Repartitioning appliance data |", "----------------------------")
    print("----------------------------", f"|             done              |", "----------------------------\n")

    print("\n----------------------------", f"|      Aggregating the data      |", "----------------------------\n")

    # Reads the data
    data = dd.read_parquet(SAVE_PATH + "final_appliance/")

    print(data.drop_duplicates(subset=['dataid'])['dataid'].values.compute())

    # Creates a dataframe containing the total consumption at a given time by one house.
    household = data[['dataid', TIME_COLUMN_NAME]].assign(
        total=data[[x for x in SENSOR_NAMES if x not in ["solar", "solar2", "battery1", "grid"]]].sum(axis=1) - data[
            ["solar", "solar2", "battery1"]].sum(axis=1))

    del data                                                            # Deletes a no longer needed dataframe

    # Reads the metadata
    metadata = dd.read_csv(
        CUSTOM_METADATA,
        dtype=METADATA_TYPE,
        blocksize=10e7,
        usecols=['dataid', 'community']
    )

    # Creates a dataframe for communities where households now have community and city information
    household = household.merge(metadata, how='left', on=["dataid"])

    # Saves a household with columns 'dataid', time, total, community
    dd.to_parquet(
        household.repartition(partition_size="100MB"),
        SAVE_PATH + "final_household/",
        write_index=False,
        partition_on=["dataid"],
        name_function=lambda x: f"data-{x}.parquet",
        schema={TIME_COLUMN_NAME: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()}
    )

    del metadata                                                        # Deletes a no longer needed dataframe
    # del household                                                       # Deletes a no longer needed dataframe

    # Creates a dataframe that has the aggregate of every community as only column.
    total_community = household.groupby(['community', TIME_COLUMN_NAME])['total'].sum()

    # Create a dataframe with columns 'time', community, total
    community = household.drop(columns=['total', 'dataid'])\
        .drop_duplicates(subset=['community', TIME_COLUMN_NAME])\
        .join(total_community, on=['community', TIME_COLUMN_NAME])

    # Saves a community with columns time, 'community', total
    dd.to_parquet(
        community.repartition(partition_size="100MB"),
        SAVE_PATH + "final_community/",
        write_index=False,
        partition_on=["community"],
        name_function=lambda x: f"data-{x}.parquet",
        schema={TIME_COLUMN_NAME: pa.timestamp(unit='s', tz='UTC'), 'community': pa.int32()}
    )

    del total_community                                                 # Deletes a no longer needed dataframe
    del community                                                       # Deletes a no longer needed dataframe

    print("\n----------------------------", f"|   Aggregating the data done    |", "----------------------------\n")

    print("\n----------------------------", f"|      Normalizing the data      |", "----------------------------\n")

    print("\n----------------------------", f"|     Normalizing households     |", "----------------------------\n")

    data = dd.read_parquet(SAVE_PATH + "final_household/")              # Reads the data from a household

    mean = data["total"].mean()                                         # The mean value of the house.
    std = data["total"].std()                                           # The standard deviation of the house.

    data["total"] = (data["total"] - mean) / std

    dd.to_parquet(
        data.repartition(partition_size="100MB"),
        SAVE_PATH + "normalized/final_household/",
        write_index=False,
        partition_on=["dataid"],
        name_function=lambda x: f"data-{x}.parquet",
        schema={TIME_COLUMN_NAME: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()}
    )

    print("\n----------------------------", f"|    Normalizing communities     |", "----------------------------\n")

    data = dd.read_parquet(SAVE_PATH + "final_community/")              # Reads the data from a community

    data.to_csv("test.csv", single_file=True)

    mean = data["total"].mean()                                         # The mean value of the community
    std = data["total"].std()                                           # The standard deviation of the community

    data["total"] = (data["total"] - mean) / std                        # Normalize the data

    dd.to_parquet(
        data.repartition(partition_size="100MB"),
        SAVE_PATH + "normalized/final_community/",
        write_index=False,
        partition_on=["community"],
        name_function=lambda x: f"data-{x}.parquet",
        schema={TIME_COLUMN_NAME: pa.timestamp(unit='s', tz='UTC'), 'community': pa.int32()}
    )

    print("\n----------------------------", f"|   Normalizing the data done    |", "----------------------------\n")

    print("\n----------------------------", f"|      Creating final file       |", "----------------------------\n")

    household = dd.read_parquet(SAVE_PATH + "normalized/final_household")
    household['community'] = household['community'].astype(dtype=np.int32)
    household = household.categorize(columns=['community'])
    household = household.rename(columns={"total": "total_household"})

    community = dd.read_parquet(SAVE_PATH + "normalized/final_community", usecols=[TIME_COLUMN_NAME, 'community', 'total'])
    community = community.rename(columns={"total": "total_community"})

    final = household.merge(community, how='left', left_on=[TIME_COLUMN_NAME, 'community'], right_on=[TIME_COLUMN_NAME, 'community'])

    dd.to_parquet(
        final.repartition(partition_size="100MB"),
        SAVE_PATH + "normalized/final_mixed/",
        write_index=False,
        partition_on=["dataid"],
        name_function=lambda x: f"data-{x}.parquet",
        schema={TIME_COLUMN_NAME: pa.timestamp(unit='s', tz='UTC'), 'dataid': pa.int32()}
    )

    # final = final[['total_community', 'total_household']]
    #
    # final.compute().to_csv("test.csv", index=False)

    print("\n----------------------------", f"|    Creating final file done    |", "----------------------------\n")

    print("\n----------------------------", f"|       Splitting the data       |", "----------------------------\n")

    dataids = final['dataid'].drop_duplicates().compute().values.tolist()
    dataids.sort()

    del final

    total = len(dataids)

    if not os.path.isdir(SAVE_PATH + "final/"):
        os.mkdir(SAVE_PATH + "final/")

    for progress, dataid in enumerate(dataids):
        print("\n----------------------------", f"|  Starting with dataid = {dataid}  |",
              "----------------------------")
        print("----------------------------", f"|        progress: {progress}/{total}         |",
              "----------------------------\n")

        # Reads the data related to one dataid
        dataid_ddf = dd.read_parquet(
            SAVE_PATH + "normalized/final_mixed",
            filters=[('dataid', '==', dataid)],
            columns=[TIME_COLUMN_NAME, 'total_community', 'total_household']
        ).compute()

        dataid_ddf.sort_values(TIME_COLUMN_NAME, ascending=True)

        length = len(dataid_ddf)

        train_index = int(length * TRAIN_SIZE)
        val_index = int(length * (TRAIN_SIZE + VAL_SIZE))

        train = dataid_ddf[0:train_index][['total_community', 'total_household']]
        val = dataid_ddf[train_index:val_index][['total_community', 'total_household']]
        test = dataid_ddf[val_index:][['total_community', 'total_household']]

        if not os.path.isdir(SAVE_PATH + f"final/house-{dataid}/"):
            os.mkdir(SAVE_PATH + f"final/house-{dataid}/")

        train.to_csv(SAVE_PATH + f"final/house-{dataid}/house-{dataid}_training_.csv", index=False)
        val.to_csv(SAVE_PATH + f"final/house-{dataid}/house-{dataid}_validation_.csv", index=False)
        test.to_csv(SAVE_PATH + f"final/house-{dataid}/house-{dataid}_test_.csv", index=False)

        print("\n----------------------------", f"|    Done with dataid = {dataid}     |",
              "----------------------------\n")

    print("\n----------------------------", f"|       Splitting the done       |", "----------------------------\n")


if __name__ == '__main__':
    main()
