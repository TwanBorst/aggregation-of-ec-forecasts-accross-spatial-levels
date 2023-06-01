from data_utils import *
from cnn_model import *
import pandas as pd
from one_shot_learning import one_shot_learning
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

# reading data
df = pd.read_csv(NEW_YORK_1MIN)
metadata_df = pd.read_csv(METADATA)
austin_df = pd.read_csv(AUSTIN_1MIN)

data, labels = prepare_data(df, metadata_df)
model = train_model(data, labels)

austin_df_trim = austin_df[austin_df["dataid"] == austin_df.loc[0, "dataid"]]
data_austin, labels_austin = prepare_data(austin_df_trim, metadata_df)

# use the first row of the austin data for few-shot learning
new_model = one_shot_learning(model, data_austin[:1], labels_austin[:1])

# evaluate the performance of the one-shot-learning using data from the same house, as the one-shot learning data
loss, accuracy = new_model.evaluate(data_austin[1:], labels_austin[1:], verbose=0)
print(f'Accuracy: {accuracy * 100}')
