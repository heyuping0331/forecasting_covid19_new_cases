


import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX

# grid search --> walk-forward-validated mape
def walk_forward_validation(p,q,d,P,Q,D,s, freq, training_range, forecast_horizon, data):

  # make grid
  pdq = list(itertools.product(p, d, q, P, D, Q, s))
  df_params = pd.DataFrame(pdq)
  df_params.columns = ['p','d','q','P','D','Q', 's']
  df_params['mape'] = None

  # starting day for each validation set
  day_list = []
  i=0
  while (len(data)-freq*i>=training_range):
      day_list.append(freq*i)
      i += 1

  # iterate over each hyperparameter combination
  for index, row in df_params.iterrows():

      mape_list = []
      # walk-forward validation
      for day in [0, freq, freq*2, freq*3, freq*4]:
          # fit model
          model = SARIMAX(data.iloc[day:day+training_range,:], order = (row['p'],row['d'],row['q']), seasonal_order = (row['P'],row['D'],row['Q'],row['s']))
          results = model.fit(max_iter = 50, method = 'powell')
          # dynamic forecast (not one-step-ahead forecast)
          pred = results.get_forecast(steps = forecast_horizon, dynamic=True).predicted_mean.reset_index()
          y = data.iloc[day+training_range:day+training_range+forecast_horizon,:].reset_index()
          temp = pd.merge(pred, y, how='inner', left_on='index', right_on='Date')
          # avg mape over 7-day horizon
          mape = np.mean(np.abs(temp[0] - temp['Cases_new'])/temp['Cases_new'])
          # n validation sets
          mape_list.append(mape)
      
      # avg avg mape over 5 validation sets
      df_params.loc[index, 'mape'] = np.mean(mape_list)

  return df_params.sort_values('mape')










