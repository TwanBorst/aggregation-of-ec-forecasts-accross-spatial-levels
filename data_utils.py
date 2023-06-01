from sklearn.preprocessing import MultiLabelBinarizer

from constants import *

import numpy as np


# function that returns the measured devices
def get_measured_devices(metadata, data_id):
    filtered_df = metadata.loc[metadata['dataid'] == str(data_id), 'air1':'wellpump1']
    filtered_df = filtered_df.columns[filtered_df.eq('yes').any()]
    measured_devices = list(filtered_df.values)
    if "grid" in measured_devices:
        measured_devices.remove("grid")
    if "solar" in measured_devices:
        measured_devices.remove("solar")
    return np.array(measured_devices)


"""
This function takes a list that contains a list of labels i.e. [["kitchenapp1", light], ["kettle", "stove]]
This is transformed to its binary equivalent 
"""
def label_binarizer(label_arr):
    bin_vector_arr = []
    for label in label_arr:
        bin_vector = np.zeros(len(SENSOR_NAMES))
        for device in label:
            bin_vector[SENSOR_NAMES.index(device)] = 1
        bin_vector_arr.append(bin_vector)
    return bin_vector_arr


def prepare_data(df, metadata_df):
    # TODO: data normalization
    data = []
    labels = []
    for data_id in df['dataid'].unique():

        # Select data for the current dataid
        data_subset = df[df['dataid'] == data_id]

        # filter the metadata dataframe
        features = get_measured_devices(metadata_df, data_id)

        # Split the data into parts of 1 hour
        num_parts = int(len(data_subset) / 60)  # Assuming 60 measurements per hour
        for i in range(num_parts):
            start_index = i * 60
            end_index = (i + 1) * 60

            # Extract the subset of data for the current part
            subset = data_subset.iloc[start_index:end_index]

            # Check which devices are on during this part
            devices_on = []
            for device in features:
                # TODO: is this a good metric?
                if subset[device].max() > THRESHOLD_ON:
                    devices_on.append(device)

            # print("num of measured devices", len(features))
            # print("num of devices on", len(devices_on))
            # replace NaN values and take sum
            subset.fillna(0, inplace=True)
            subset.loc[:, "sum"] = subset[features].sum(axis=1)

            data.append(subset["sum"].values)
            labels.append(devices_on)

        # Convert the labels into binary encoded vectors
        y = label_binarizer(labels)

        # Convert the data and labels to NumPy arrays
        data = np.array(data)
        y = np.array(y)

        # Reshape the input data for CNN
        data = data.reshape((data.shape[0], data.shape[1]))

        return data, y
