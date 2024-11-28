# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def linear_regression(x, y):
    
    model = LinearRegression().fit(x, y)

    # Predict the next value
    next_time = np.array([[30]])  # Next time step
    predicted_price = model.predict(next_time)
    
    return predicted_price

def moving_average(y, window_size):

    predicted_ma = np.mean(y[-window_size:]) 
    
    return predicted_ma


def simple_exp_smoothing(y, alpha):
    
    exp_model = SimpleExpSmoothing(y).fit(smoothing_level = alpha, optimized=False)

    predicted_exp = exp_model.forecast(1)
    
    return predicted_exp

def holt_winters(y):

    hw_model = ExponentialSmoothing(y, trend = "add", seasonal = None).fit()
    predicted_hw = hw_model.forecast(1)  # Forecast the next value

    return predicted_hw