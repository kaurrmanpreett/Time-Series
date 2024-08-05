from modules.imports import *
from modules.data_preparation import *
from modules.model_training import *
from modules.model_evaluation import *


file_path = 'C:\\Users\\kaurr\\OneDrive\\Desktop\\BISI\\2208\\Time Series\\data\\AAPL.csv'
data = load_data(file_path)
df = pre_prep_data(data)
check_data(df)
decomposition_analysis(df)
ar_model = train_arimamodel(df)
ypred,conf_int = forecast_using_Arima(ar_model)
dataframe_from_prediction_values(data,ypred,conf_int)
dfx = Bivariate_using_ExogenousVariable(data)
arimax = train_arimamodel_bivariate(dfx)
ex,ypred,conf_int = evaluate_arima_bivariate(arimax,data)
dataframe_from_predvalues_ex(data,ypred,conf_int)
dataYF = yfinance_dataset_API()
train,test,model1,model1_preds,features = train_XGBOOST(dataYF)
evaluation_XGBOOST(test,model1_preds)