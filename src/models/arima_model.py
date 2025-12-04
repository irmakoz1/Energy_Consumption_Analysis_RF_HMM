from statsmodels.tsa.arima.model import ARIMA

def train_arima(series, order=(3, 1, 3)):
    model = ARIMA(series, order=order)
    model = model.fit()
    return model