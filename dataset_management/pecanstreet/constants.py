import pandas as pd

# New York Data
NEW_YORK_1S = "/Users/stan/Documents/uni/year_3/rp/seq2point-nilm-master/data/pecanstreet/1s_data_newyork_file1.csv"
NEW_YORK_1MIN = "/Users/stan/Documents/uni/year_3/rp/seq2point-nilm-master/data/pecanstreet/1minute_data_newyork.csv"
NEW_YORK_15MIN = "/Users/stan/Documents/uni/year_3/rp/seq2point-nilm-master/data/pecanstreet/15minute_data_newyork.csv"

# California Data
CALIFORNIA_1S = ""
CALIFORNIA_1MIN = ""
CALIFORNIA_15MIN = ""

# Austin Data
AUSTIN_1S = ""
AUSTIN_1MIN = ""
AUSTIN_15MIN = ""

# From and end time that will be used throughout the code
FROM_TIME = pd.Timestamp('2018-06-01 05:00:00+00:00', tz="UTC")
END_TIME = pd.Timestamp('2018-09-01 04:45:00+00:00', tz="UTC")

# Some file specific data
METADATA_TIME_PREFIX = 'egauge_1min_'
TIME_COLUMN_NAME = 'local_15min'

# Metadata
METADATA = "/Users/stan/Documents/uni/year_3/rp/seq2point-nilm-master/data/pecanstreet/metadata.csv"
METADATA_TYPE = {
    'dataid': int,
    'active_record': bool,
    'building_type': str,
    'city': str,
    'state': str,
    METADATA_TIME_PREFIX + 'min_time': object,
    METADATA_TIME_PREFIX + 'max_time': object
}
COMMUNITY_SIZE = 5

# Directories
DATA_PATH = "/Users/stan/Documents/uni/year_3/rp/seq2point-nilm-master/data/pecanstreet/"
CUSTOM_METADATA = '/Users/stan/Documents/uni/year_3/rp/seq2point-nilm-master/data/pecanstreet/15minute/metadata_15minute.csv'
SAVE_PATH = '/Users/stan/Documents/uni/year_3/rp/seq2point-nilm-master/data/pecanstreet/15minute/'
DASK_TMP_DIR = '/Users/stan/Documents/uni/year_3/rp/seq2point-nilm-master/tmp/'

# Statistics used for normalization
AGG_MEAN = 522
AGG_STD = 814

# Timestep frequency
TIME_FREQUENCY = "1min"

# Columns
# FEATURE_COLUMNS = ['dayofyear', 'season', 'month', 'dayofweek', 'hourofday', 'minuteofhour', 'typeofday']
SENSOR_NAMES = ["air1", "air2", "air3", "airwindowunit1", "aquarium1", "bathroom1", "bathroom2", "bedroom1", "bedroom2",
                "bedroom3", "bedroom4", "bedroom5", "battery1", "car1", "car2", "circpump1", "clotheswasher1",
                "clotheswasher_dryg1", "diningroom1", "diningroom2", "dishwasher1", "disposal1", "drye1", "dryg1",
                "freezer1", "furnace1", "furnace2", "garage1", "garage2", "grid", "heater1", "heater2", "heater3",
                "housefan1", "icemaker1", "jacuzzi1", "kitchen1", "kitchen2", "kitchenapp1", "kitchenapp2",
                "lights_plugs1", "lights_plugs2", "lights_plugs3", "lights_plugs4", "lights_plugs5", "lights_plugs6",
                "livingroom1", "livingroom2","microwave1", "office1", "outsidelights_plugs1", "outsidelights_plugs2",
                "oven1", "oven2", "pool1", "pool2", "poollight1", "poolpump1", "pump1", "range1", "refrigerator1",
                "refrigerator2", "security1", "sewerpump1", "shed1", "solar", "solar2", "sprinkler1", "sumppump1",
                "utilityroom1", "venthood1", "waterheater1", "waterheater2", "wellpump1", "winecooler1"]

# Split size
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1


