


import os
import pandas as pd 
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics

# add conditional seasonality
def fit_model(data, trend_func, changepoint_prior, changepoint_range, weekly_fourier_order, seasonal_mode):
    '''

    Output --> FB Prophet object
    '''
    m = Prophet(
        growth = trend_func
        ,weekly_seasonality = True
        ,yearly_seasonality = False
        ,daily_seasonality = False
        ,changepoint_prior_scale = changepoint_prior
        ,changepoint_range = changepoint_range
        ,seasonality_mode = seasonal_mode)

    if trend_func == 'logistic':
        data['cap'] = 12000
        data['floor'] = 10

    return m.fit(data)


def cross_validate(fitted_model, training_range, forecast_range, cv_interval):   
    '''
    Input --> 1 already defined model
    Output --> avg MAPE
    '''
    # nee dot make sure that each initial, horizon covers full week??
    df_cv = cross_validation(fitted_model, initial=training_range, horizon=forecast_range, period=cv_interval)
    df_p = performance_metrics(df_cv, rolling_window = 1/7)
    
    return df_p['mape'].mean()


def tune_hyperparameters(data, model_prophet, model_cv, grid):
    '''
    Input --> 

    Output --> best parameters
    '''

    # make grid
    df1 = pd.DataFrame({'growth': grid['growth']})
    df2 = pd.DataFrame({'changepoint_prior': grid['changepoint_prior']})
    df3 = pd.DataFrame({'changepoint_range': grid['changepoint_range']})
    df4 = pd.DataFrame({'weekly_fourier_order': grid['weekly_fourier_order']})
    df5 = pd.DataFrame({'seasonal_mode': grid['seasonal_mode']})
    df_param = df1.assign(foo=1).merge(df2.assign(foo=1)).merge(df3.assign(foo=1)).merge(df4.assign(foo=1)).merge(df5.assign(foo=1)).drop('foo', 1)
    df_param['MAPE'] = None

    # get MAPE for each parameter set
    for row in range(len(df_param)):       
        model = model_prophet(data, 
            df_param.loc[row, 'growth'], 
            df_param.loc[row, 'changepoint_prior'], 
            df_param.loc[row, 'changepoint_range'], 
            df_param.loc[row, 'weekly_fourier_order'], 
            df_param.loc[row, 'seasonal_mode'])
        mape = model_cv(model, '21 days', '7 days', '7 days')

        df_param.loc[row,'MAPE'] = mape

    return df_param.sort_values('MAPE')


def make_forecasts(model_prophet, ts, horizon,
    best_trend_func, best_changepoint_prior, best_changepoint_range, best_weekly_fourier_order, best_seasonal_mode):
    '''

    Output --> final_model, final_forecasts
    '''

    # fit final model
    model = model_prophet(
        data = ts, 
        trend_func = best_trend_func, 
        changepoint_prior = best_changepoint_prior, 
        changepoint_range = best_changepoint_range, 
        weekly_fourier_order = best_weekly_fourier_order,
        seasonal_mode = best_seasonal_mode)
    
    # create future dates
    ts_future = model.make_future_dataframe(periods = horizon)

    if best_trend_func == 'logistic':
        ts_future['cap'] = 12000
        ts_future['floor'] = 0

    # return predictions
    ts_forecast = model.predict(ts_future)

    # decomposition


    # plot final forecasts

    return model, ts_forecast