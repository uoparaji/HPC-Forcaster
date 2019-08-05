def run_forecast():
	from forecast import lstm_forecast
	lstm_forecast(91, "20181007002224.h5", "test_set_size=2219_week=51.csv", test_model=True)
    
    
def run_forecast_bin():
	from forecast_01 import lstm_forecast_bin
	lstm_forecast_bin(7, "01-20181010103830.h5", "test_set_size=312_week=8.csv", test_model=True)