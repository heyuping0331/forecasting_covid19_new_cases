
import os
import pandas as pd 
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics

def fit_model(data, trend_func, changepoint_prior, changepoint_range, weekly_fourier_order):
    '''

    Output --> FB Prophet object
    '''
    m = Prophet(
        growth = trend_func
        ,weekly_seasonality = True
        ,yearly_seasonality = False
        ,daily_seasonality = False
        ,changepoint_prior_scale = changepoint_prior
        ,changepoint_range = changepoint_range)

    if trend_func == 'logistic':
        data['cap'] = 12000
        data['floor'] = 0

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

# mape = cross_validate(test, '21 days', '7 days', '7 days')

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
    df_param = df1.assign(foo=1).merge(df2.assign(foo=1)).merge(df3.assign(foo=1)).merge(df4.assign(foo=1)).drop('foo', 1)
    df_param['MAPE'] = None

    # get MAPE for each run
    for row in range(len(df_param)):       
        model = model_prophet(data, df_param.loc[row, 'growth'], df_param.loc[row, 'changepoint_prior'], df_param.loc[row, 'changepoint_range'], df_param.loc[row,'weekly_fourier_order'])
        mape = model_cv(model, '21 days', '7 days', '7 days')

        df_param.loc[row,'MAPE'] = mape

    return df_param.sort_values('MAPE')


# my_grid = {
#     'growth': ['linear', 'logistic']
#     ,'changepoint_prior': [0.01, 0.1, 0.5]
#     ,'changepoint_range': [0.8]
#     ,'weekly_fourier_order': [3, 5]
# }

# tune_hyperparameters(data=ts_ny_train, model_prophet=fit_model, model_cv=cross_validate, grid=my_grid


def make_forecasts(model_prophet, best_trend_func, best_changepoint_prior, best_changepoint_range, best_weekly_fourier_order, ts, horizon):
    '''

    Output --> final_model, final_forecasts
    '''

    # fit final model
    model = model_prophet(
        data = ts, 
        trend_func = best_trend_func, 
        changepoint_prior = best_changepoint_prior, 
        changepoint_range = best_changepoint_range, 
        weekly_fourier_order = best_weekly_fourier_order)
    
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