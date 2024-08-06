from .imports import *

def load_data(file_path):
    # Loads data from file.
    data = pd.read_csv(file_path)
    return data

def pre_prep_data(data):
    # Prepares data for plotting. 
    print(f"Datatypes are: ", data.dtypes)
    data['Date'] = pd.to_datetime(data['Date'])
    df = data.iloc[:-2,0:2]
    df = df.set_index('Date')
    return df
    
def check_data(df): 
    
    # Check data and show seaborn lineplot if data is stationary.
    print(f"Datatypes After Change: ", df.dtypes)
    #create seaborn lineplot
    plot = sns.lineplot(df['AAPL'])

    #rotate x-axis labels
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    plt.show()
    # if p value < 0.05 the series is stationary
    results = adfuller(df['AAPL'])
    print('p-value:', results[1]) # adf, pvalue, usedlag_, nobs_, critical_values_, icbest_

def making_data_stationery(df):
    # Plots 1st order differencing of AAPL. The plot is based on adfuller function
    # 1st order differencing
    v1 = df['AAPL'].diff().dropna()
    # adf test on the new series. if p value < 0.05 the series is stationary
    results1 = adfuller(v1)
    print('p-value:', results1[1]) # adf, pvalue, usedlag_, nobs_, critical_values_, icbest_
    plt.plot(v1)
    plt.title('1st order differenced series')
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Price (USD)')
    plt.show()

def Bivariate_using_ExogenousVariable(data):
    # Takes data from Bivariate_using_ExogenousVariable and converts it to a DataFrame
    dfx = data.iloc[0:-2,0:3]
    return dfx

def yfinance_dataset_API():
    # Download and make Yfinance dataset using API.
    dataYF = yf.download("AAPL", start="2000-01-01", end="2022-05-31")
    dataYF['Next_day'] = dataYF['Close'].shift(-1)
    dataYF['Target'] = (dataYF['Next_day'] > dataYF['Close']).astype(int)
    dataYF.head(5)
    from matplotlib import pyplot as plt
    dataYF['Close'].plot(kind='line', figsize=(8, 4), title='line Plot')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.show()
    print("Yfinance dataset make success")
    return dataYF
