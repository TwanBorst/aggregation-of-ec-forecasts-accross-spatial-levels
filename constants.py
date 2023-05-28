import pandas as pd

# Input and output timesteps for the models
INPUT_SIZE = 672
OUTPUT_SIZE = 96

# From and end time that will be used thoughout the code
FROM_TIME = pd.Timestamp('2019-05-01 05:00:00+00:00', tz="UTC")
END_TIME = pd.Timestamp('2019-11-01 04:45:00+00:00', tz="UTC")

# Timestep frequency
TIME_FREQUENCY = "15min"

# KFold and test fraction configuration for splitting the data
FOLDS = 5
TEST_FRACTION = 0.2

# File and folder paths
ORIGINAL_METADATA = '../data/metadata.csv'
CUSTOM_METADATA = '../data/generated/metadata_15min.csv'
DATA_GLOB = '../data/15minute_data*/15minute_data*.csv'
SAVE_DIR = 'D:/RP_Data/15min/'      # A lot of data can be stored in this folder depending on the time frequency and size of the data!
DASK_TMP_DIR = 'D:/tmp'             # Dask will use this folder when it runs out of memory or when a shuffle needs to take place. A lot of data can temporarily be stored in this folder!


METADATA_TIME_PREFIX = 'egauge_1min_'
TIME_COLUMN_NAME = 'local_15min'

# No need to change anything here
FEATURE_COLUMNS = ['dayofyear', 'season', 'month', 'dayofweek', 'hourofday', 'minuteofhour', 'typeofday']
SENSOR_NAMES = ["air1", "air2", "air3", "airwindowunit1", "aquarium1", "bathroom1", "bathroom2", "bedroom1", "bedroom2", "bedroom3", "bedroom4", "bedroom5", "battery1",
                "car1", "car2", "circpump1", "clotheswasher1", "clotheswasher_dryg1", "diningroom1", "diningroom2", "dishwasher1", "disposal1", "drye1", "dryg1", "freezer1",
                "furnace1", "furnace2", "garage1", "garage2", "grid", "heater1", "heater2", "heater3", "housefan1", "icemaker1", "jacuzzi1", "kitchen1", "kitchen2",
                "kitchenapp1", "kitchenapp2", "lights_plugs1", "lights_plugs2", "lights_plugs3", "lights_plugs4", "lights_plugs5", "lights_plugs6", "livingroom1", "livingroom2",
                "microwave1", "office1", "outsidelights_plugs1", "outsidelights_plugs2", "oven1", "oven2", "pool1", "pool2", "poollight1", "poolpump1", "pump1", "range1",
                "refrigerator1", "refrigerator2", "security1", "sewerpump1", "shed1", "solar", "solar2", "sprinkler1", "sumppump1", "utilityroom1", "venthood1", "waterheater1",
                "waterheater2", "wellpump1", "winecooler1"]
