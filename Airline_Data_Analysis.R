library(tidyverse)
library(lubridate)
library(modeltime)
library(timetk)
library(skimr)
library(h2o)
library(caTools)
library(highcharter)
library(tidymodels)
library(zoo)

#1. Using arima_boost(), exp_smoothing(), prophet_reg() models
raw<-read.csv('AirPassengers (3).csv')
colnames(raw)<-c('Date', 'Count')
#data is given monthly 
raw$Date<-as.Date(as.yearmon(raw$Date))
raw$Count<-as.numeric(raw$Count)

raw %>% plot_time_series(Date, Count);

# Train-Test split woth 80/20 ratio
splits <- initial_time_split(raw, prop = 0.8)

#boosted ARIMA using arima_boost()
model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015
) %>%
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(Count ~ Date + as.numeric(Date) + factor(lubridate::month(Date, label = TRUE), ordered = F),
      data = training(splits))

#Error-Trend-Season (ETS) model using an Exponential Smoothing State Space model
model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(Count ~ Date, data = training(splits))

#Prophet ----
model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(Count ~ Date, data = training(splits))

#Adding built models to a Model Table.----
models_tbl <- modeltime_table(
  model_fit_arima_boosted,
  model_fit_ets,
  model_fit_prophet)

#Calibrating the model to a testing set----
calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

#Testing Set Forecast & Accuracy Evaluation----
#a. Visualizing the Forecast Test with all built models on same plot----
calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = raw
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, .interactive=TRUE # For mobile screens
  )

#b. 2. Accuracy Metrics: comparing RMSE socres on test set
calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy()

#best is arima's boosted model in terms of RMSE score, we will further go with it

#3. Making forecast on lowest RMSE score model----
#Prediction for the new data
new_data_n <- seq(as.Date("1961-01-01"), as.Date("1961-12-01"), "month") %>%
  as_tibble() %>% 
  add_column(Count=0) %>% 
  rename(Date=value) %>% 
  tk_augment_timeseries_signature() %>%
  select(-contains("hour"),
         -contains("day"),
         -contains("week"),
         -minute,-second,-am.pm) %>% 
  mutate_if(is.ordered, as.character) %>% 
  mutate_if(is.character,as_factor)

#predictions for the next year
new_predictions <- model_fit_arima_boosted %>%  modeltime_calibrate(new_data = new_data_n) %>% 
  modeltime_forecast(new_data_n) %>% 
  as_tibble() %>%
  add_column(Date=new_data_n$Date) %>% 
  select(Date,.value) %>% 
  rename(Count=.value)

#4. Visualizing past data and forecast values on one plot; making separation with two different colors----
# red for past data, green for prediction
raw %>% 
  bind_rows(new_predictions) %>% 
  mutate(categories=c(rep('Actual',nrow(raw)),rep('Predicted',nrow(new_predictions)))) %>% 
  hchart("line", hcaes(Date, Count, group = categories)) %>% 
  hc_title(text='Forecast the for next year') %>% 
  hc_colors(colors = c('red','green'))

