from .imports import *
from .data_preparation import *
from .model_training import *

def forecast_using_Arima(ar_model):
    # Use Arima to forecast the time series. This is a wrapper for get_forecast
    forecast = ar_model.get_forecast(2)
    ypred = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    print(ypred)
    return ypred,conf_int
    
    
def dataframe_from_prediction_values(data,ypred,conf_int):
    # Create dataframes from predicted values
    Date = pd.Series(['2024-01-01', '2024-02-01'])
    price_actual = pd.Series(['184.40','185.04'])
    price_predicted = pd.Series(ypred.values)
    lower_int = pd.Series(conf_int['lower AAPL'].values)
    upper_int = upper_series = pd.Series(conf_int['upper AAPL'].values)

    dp = pd.DataFrame([Date, price_actual, lower_int, price_predicted, upper_int], index =['Date','price_actual', 'lower_int', 'price_predicted', 'upper_int']).T
    dp = dp.set_index('Date')
    dp.index = pd.to_datetime(dp.index)
    print("Dataframe Creation from predicted values successful")
    print(dp)
    data = data.set_index('Date')
    plt.plot(data.AAPL)
    plt.plot(dp.price_predicted, color='orange')
    plt.fill_between(dp.index,
                    lower_int,
                    upper_int,
                    color='k', alpha=.15)


    plt.title('Model Performance')
    plt.legend(['Actual','Prediction'], loc='lower right')
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Price (USD)')
    plt.show()
    print('ARIMA MAE = ', mean_absolute_error(dp.price_actual, dp.price_predicted))
    
    
    
def evaluate_arima_bivariate(arimax,data):
    # Evaluate arima bivariate data. This is a wrapper for get_forecast
    forecast = arimax.get_forecast(steps=2)
    ypred = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    print(data.tail())
    data.TXN.iloc[-2:]
    ex = data.TXN.iloc[-2:].values
    print(ex)
    forecast = arimax.get_forecast(2, exog=ex)
    ypred = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    return ex,ypred,conf_int

def dataframe_from_predvalues_ex(data,ypred,conf_int):
    # Creates dpx dataframe containing predvalues from bivariate.
    Date = pd.Series(['2024-01-01', '2024-02-01'])
    price_actual = pd.Series(['184.40','185.04'])
    price_predicted = pd.Series(ypred.values)
    lower_int = pd.Series(conf_int['lower AAPL'].values)
    upper_int = upper_series = pd.Series(conf_int['upper AAPL'].values)

    dpx = pd.DataFrame([Date, price_actual, lower_int, price_predicted, upper_int], index =['Date','price_actual', 'lower_int','price_predicted','upper_int' ]).T
    dpx = dpx.set_index('Date')
    dpx.index = pd.to_datetime(dpx.index)
    print("succesfully created dpx dataframe containing pred values from bivariate")
    print(dpx)
    print('ARIMAX MAE = ', mean_absolute_error(dpx.price_actual, dpx.price_predicted))
    
    
def evaluation_XGBOOST(test,model1_preds):
    # Evaluate XGBOOST on test data.
    print(precision_score(test['Target'], model1_preds))
    plt.plot(test['Target'], label='Actual')
    plt.plot(model1_preds, label='Predicted')
    plt.legend()
    plt.show()
