# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import torch
import torch.nn.functional as F

def linear_regression(x, y, window_size):
    
    model = LinearRegression().fit(x, y)

    # Predict the next value
    next_time = np.array([[window_size]])  # Next time step
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


def eval_baseline(test_loader):

    total_mape_linr = 0
    total_mape_mov_avg = 0
    total_mape_exp_smoothing = 0
    total_mape_holt_winters = 0

    total_rmse_linr = 0
    total_rmse_mov_avg = 0
    total_rmse_exp_smoothing = 0
    total_rmse_holt_winters = 0    
    
    for data in test_loader:

        y = data.y

        # Compute the Mean Absolute Percentage Error
        

        mape_linr = float(torch.mean(torch.abs((y - data.linr_regr) / (y + 1e-7))))
        mape_mov_avg = float(torch.mean(torch.abs((y - data.mov_avg) / (y + 1e-7))))
        mape_exp_smoothing = float(torch.mean(torch.abs((y - data.exp_smoothing) / (y + 1e-7))))
        mape_holt_winters = float(torch.mean(torch.abs((y - data.holt_winters) / (y + 1e-7))))

        total_mape_linr += mape_linr
        total_mape_mov_avg += mape_mov_avg
        total_mape_exp_smoothing += mape_exp_smoothing
        total_mape_holt_winters += mape_holt_winters

        # Compute the Mean Square Error

        mse_linr = F.mse_loss(data.linr_regr, y)
        mse_mov_avg = F.mse_loss(data.mov_avg, y)
        mse_exp_smoothing = F.mse_loss(data.exp_smoothing, y)
        mse_holt_winters = F.mse_loss(data.holt_winters, y)

        # Compute the Root Mean Square Error

        rmse_linr = torch.sqrt(mse_linr)
        rmse_mov_avg = torch.sqrt(mse_mov_avg)
        rmse_exp_smoothing = torch.sqrt(mse_exp_smoothing)
        rmse_holt_winters = torch.sqrt(mse_holt_winters)

        total_rmse_linr += rmse_linr
        total_rmse_mov_avg += rmse_mov_avg
        total_rmse_exp_smoothing += rmse_exp_smoothing
        total_rmse_holt_winters += rmse_holt_winters

        
    mape = (total_mape_linr/len(test_loader), total_mape_mov_avg/len(test_loader), total_mape_exp_smoothing/len(test_loader), total_mape_holt_winters/len(test_loader))
    
    rmse = (total_rmse_linr/len(test_loader), total_rmse_mov_avg/len(test_loader), total_rmse_exp_smoothing/len(test_loader), total_rmse_holt_winters/len(test_loader))

    # Order: Linear Regression, Moving Average, Exponential Smoothing, Holt-Winters
    
    return mape, rmse



