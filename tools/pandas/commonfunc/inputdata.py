from os import getcwd, path
import pandas as pd


def read_data(file_name='GBPUSD_Daily.csv'):
    location_of_files = getcwd()
    file_path = path.join(location_of_files, '..', 'data', file_name)
    df = pd.read_csv(file_path)
    df['PCT'] = df.Close.pct_change()
    return df
