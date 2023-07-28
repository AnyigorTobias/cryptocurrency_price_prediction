import numpy as np
from sklearn.metrics import mean_squared_error

def MAPE(y_test, y_pred):
   MAPE= np.average(np.abs((y_test - y_pred) / y_test)) * 100
   return f"{MAPE:.2f}%"
   
def rmse(y_test, y_pred):
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse