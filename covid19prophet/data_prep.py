

import os
import pandas as pd 
import numpy as np

# load data
# os.getcwd()
os.chdir('/Users/yupinghe/Documents/Futuristic/covid19')

# county-level confirmed cases
df = pd.read_csv('time_series_covid19_confirmed_US.csv')
print(f'Dataset size: ', {df.shape}, '\n') 
# print(df.columns)
print(df.head())


# wide to long
df_ny = df.loc[df['Province_State']=='New York', df.columns[11:].to_list()].melt(
    id_vars=None, 
    value_vars=df.columns[11:].to_list(), 
    var_name='Date', 
    value_name='Cases')

df_ny['Date'] = pd.to_datetime(df_ny['Date'])

# Set to time series
ts_ny = df_ny.set_index(df_ny['Date']).loc[:,['Cases']].resample('D').sum()

# Calcualte daily new cases
ts_ny['Cases_lag1'] = ts_ny.shift(1)
ts_ny['Cases_new'] = ts_ny['Cases'] - ts_ny['Cases_lag1']

# Propeht format
ts_ny_train = ts_ny.loc['2020-03-01':,['Cases_new']].reset_index().rename(columns={'Date':'ds', 'Cases_new':'y'})
ts_ny_train.tail()

