import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error




def make_dataset(dataset, store_number, return_train_test=False, train_size=0.8):
    
    store_df = dataset[dataset['Store'] == store_number]
    
    train_size = int(len(store_df)*train_size)
        
    train = store_df.iloc[:train_size]
    test = store_df.iloc[train_size:]
    
    if return_train_test:
        return store_df, train['Weekly_Sales'], test['Weekly_Sales']
    else:
        return store_df
            
    
    
def preprocess_data(scaled_data, time_step=12):
    dataX, dataY = [], []
    
    for i in range(len(scaled_data)-time_step-1):
        a = scaled_data[i:(i+time_step), 0]
        
        dataX.append(a)
        dataY.append(scaled_data[(i+time_step)])
        
    return np.array(dataX), np.array(dataY)
    
    



class Seasonality:
    def __init__(self, df):
        self.df=df
        
        
    def seasonal_decompose(self, store_number,  col: str, period=52, return_data=False):
        
        store_df = self.df[self.df['Store'] == store_number].copy()
        decompose = seasonal_decompose(store_df[col], period=period)
        print(decompose.plot())
        plt.show()
        
    
    def adfuller(self, store_number:int, col:str, return_data=False):
        
        store_df = self.df[self.df['Store'] == store_number].copy()
        res = adfuller(store_df[col])
        
        if return_data:
            return res 
        else:
            return res[1]
    
    
class TsaPlots:
    def __init__(self, df):
        self.df = df
        
    def acf_plot(self, col:str , store_number: int):
        
        store_df = self.df[self.df['Store'] == store_number].copy()
        
        plot_acf(store_df[col])
        plt.title(f'ACF plot for {col}')
        plt.xlabel('Lags')
        plt.ylabel('ACF coefficient')
        
    def pacf_plot(self, col: str, store_number: int):
        
        store_df = self.df[self.df['Store'] == store_number].copy()
        
        plot_pacf(store_df[col])
        plt.title(f'PACF plots for s{col}')
        plt.xlabel('Lags')
        plt.ylabel('PACF coefficient')
        
    def  combined_tsa_plots(self, col:str, store_number: int):
        self.acf_plot(col, store_number)
        self.pacf_plot(col, store_number)
        plt.show()
        
        
class TimeSeriesModel:
    def __init__(self, df):
        self.df = df
    
    
    def build_arima(self, store_number: int, train_size:int, order=(1,1,1),
                   return_data=False, return_model=False):
        
        print("Selected Store : ", store_number)
        
        store_df = self.df[self.df['Store'] == store_number][['Weekly_Sales']]
        store = self.df[self.df['Store'] == store_number]
        
            
        train_size = int(len(store_df)*train_size)
        
        train = store_df.iloc[:train_size]
        test = store_df.iloc[train_size:]
            
            
        model = ARIMA(store_df, order=order)
        model_fit = model.fit()
        preds = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
        
        
        print("Mean absolute Percentage error for the ARIMA model ", mean_absolute_percentage_error(test, preds))
        print('Root Mean Squared Error for the ARIMA model: ', np.sqrt(mean_squared_error(test, preds)))
        
        
        fig, ax = plt.subplots(2, 1, figsize=(15,10))


        test.plot(label='Test Sales', ax=ax[0], color='blue')
        preds.plot(label='Predicted test Sales',ax=ax[0], color='red')
        ax[0].set_title(f'Predicted test sales vs Actual test Sales for ARIMA model of order {order}' )
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Weekly Sales')
        ax[0].legend()

        store_df.plot(label= 'Actual Sales', ax=ax[1], color='blue')
        preds.plot(label='Predicted Sales', ax=ax[1], color='red')
        ax[1].set_title(f'Actual Sales vs Predicted Sales for ARIMA model of order {order}'  )
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Weekly Sales')
        ax[1].legend()
        
        plt.tight_layout()
        
        

        if return_data:
            preds, store, train, test
        if return_model:
            return model_fit
            
        
    def forecast_arima(self, store_number: int, train_size:int, order=(1,1,1),
                       steps=12, return_data=False, return_model=False):
        
        print("Selected Store : ", store_number)
        
        store_df = self.df[self.df['Store'] == store_number][['Weekly_Sales']]
        store = self.df[self.df['Store'] == store_number]
        
            
        train_size = int(len(store_df)*train_size)
        
        train = store_df.iloc[:train_size]
        test = store_df.iloc[train_size:]
            
            
        model = ARIMA(store_df, order=order)
        model_fit = model.fit()
        preds = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
        
        forecast = model_fit.forecast(steps=steps)
        
        
        print("Mean absolute Percentage error for the ARIMA model ", mean_absolute_percentage_error(test, preds))
        print('Root Mean Squared Error for the ARIMA model: ', np.sqrt(mean_squared_error(test, preds)))
        
        
        fig, ax = plt.subplots(2, 1, figsize=(15,10))


        store_df.plot(label='Atual  Sales', ax=ax[0], color='blue')
        preds.plot(label='Predicted test Sales',ax=ax[0], color='red')
        ax[0].set_title(f'Predicted test sales vs Actual test Sales for ARIMA model of order {order}' )
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Weekly Sales')
        ax[0].legend()

        store_df.plot(label= 'Actual Sales', ax=ax[1], color='blue')
        forecast.plot(label='Forecasted Sales', ax=ax[1], color='red')
        ax[1].set_title(f'Actual Sales vs Forecasted for ARIMA model of order {order}')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Weekly Sales')
        ax[1].legend()
        
        plt.tight_layout()
        
        
        forecast_df = pd.DataFrame(index=forecast.index, data=forecast.values, columns=['Forecast'])
         
        

        if return_data:
            forecast, preds, store, train, test 
            
        if return_model:
            return model_fit
        
        else:
            return forecast_df
                
        
            
        
    

    def build_sarimax(self, store_number: int, train_size=0.8, order=(1,1,1),
                    seasonal_order=(1,1,1,52), return_data=False, return_model=False):
        
        print("Selected Store : ", store_number)
        
        store_df = self.df[self.df['Store'] == store_number][['Weekly_Sales']]
        store = self.df[self.df['Store'] == store_number]
        
            
        train_size = int(len(store_df)*train_size)
        
        train = store_df.iloc[:train_size]
        test = store_df.iloc[train_size:]
            
            
        model = SARIMAX(store_df, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        preds = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
        
        
        print("Mean absolute Percentage error for the SARIMAX model ", mean_absolute_percentage_error(test, preds))
        print('Root Mean Squared Error for the SARIMAX model: ', np.sqrt(mean_squared_error(test, preds)))
        
        
        fig, ax = plt.subplots(2, 1, figsize=(15,10))


        test.plot(label='Test Sales', ax=ax[0], color='blue')
        preds.plot(label='Predicted test Sales',ax=ax[0], color='red')
        ax[0].set_title(f'Predicted test sales vs Actual test Sales for SARIMAX model of order {order}' )
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Weekly Sales')
        ax[0].legend()

        store_df.plot(label= 'Actual Sales', ax=ax[1], color='blue')
        preds.plot(label='Predicted Sales', ax=ax[1], color='red')
        ax[1].set_title(f'Actual Sales vs Predicted Sales for SARIMAX model of order {order}'  )
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Weekly Sales')
        ax[1].legend()
        
        plt.tight_layout()
        
        

        if return_data:
            preds, store, train, test
        if return_model:
            return model_fit    
        
        
    def forecast_sarimax(self, store_number: int, train_size:int, order=(1,1,1),
                         seasonal_order=(1,1,1,52), steps=12, 
                         return_data=False, return_model=False):
        
        print("Selected Store : ", store_number)
        
        store_df = self.df[self.df['Store'] == store_number][['Weekly_Sales']]
        store = self.df[self.df['Store'] == store_number]
        
            
        train_size = int(len(store_df)*train_size)
        
        train = store_df.iloc[:train_size]
        test = store_df.iloc[train_size:]
            
            
        model = SARIMAX(store_df, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        preds = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
        
        forecast = model_fit.forecast(steps=steps)
        
        
        print("Mean absolute Percentage error for the ARIMA model ", mean_absolute_percentage_error(test, preds))
        print('Root Mean Squared Error for the ARIMA model: ', np.sqrt(mean_squared_error(test, preds)))
        
        
        fig, ax = plt.subplots(2, 1, figsize=(15,10))


        store_df.plot(label='Atual  Sales', ax=ax[0], color='blue')
        preds.plot(label='Predicted test Sales',ax=ax[0], color='red')
        ax[0].set_title(f'Predicted test sales vs Actual test Sales for ARIMA model of order {order}' )
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Weekly Sales')
        ax[0].legend()

        store_df.plot(label= 'Actual Sales', ax=ax[1], color='blue')
        forecast.plot(label='Forecasted Sales', ax=ax[1], color='red')
        ax[1].set_title(f'Actual Sales vs Forecasted Sales for SARIMAX model of order {order}')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Weekly Sales')
        ax[1].legend()
        
        plt.tight_layout()

        
        forecast_df = pd.DataFrame(index=forecast.index, data=forecast.values, columns=['Forecast'])
        
    
        if return_data:
            forecast, preds, store, train, test 
            
        if return_model:
            return model_fit
        
        else:
            return forecast_df