import pandas as pd


def load_dataframe(directory, building, channel, col_names=['time', 'data'], nrows=None):
    """

    Args:
        directory:  The path to the file you want to read from.
        building:   The house for which you want to read the data.
        channel:    The channel that correspond to the appliance you want to train for.
        col_names:
        nrows:

    Returns:

    """
    df = pd.read_table(directory + 'house_' + str(building) + '/' + 'channel_' +
                       str(channel) + '.dat',
                       sep="\s+",
                       nrows=nrows,
                       usecols=[0, 1],
                       names=col_names,
                       dtype={'time': str},
                       )
    return df