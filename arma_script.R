library(forecast)
library(tseries)
library(survMisc)
#library(bayesforecast)
library(magrittr)
library(zoo)

# als <- c(10, 30, 100, 300, 2070)
als <- c(10)

for (al in als) {
  data <- read.csv(sprintf('C:\\Users\\lenna\\Documents\\Delft\\Y3\\RP\\Graph-WaveNet\\data\\2070HH_al%d.csv', al))
  num_houses <- ncol(data)
  
  prediction_list <- list()
  
  for (i in 2:num_houses) {
    print(i)
    house_data <- ts(data[, i], start = c(2013,01,01),frequency = 8760)
    house_data <- SplitData(house_data)
    diff_train <- diff(x=house_data$train, lag=24)
    
    arma_model <- arima(diff_train, order=c(2, 1, 1))
    
    
    forecasted <- forecast(arma_model)
    prediction_list[[i]] <- forecasted
  }
}


SplitData <- function(ts_data) {
  ret <- list()
  
  idx <- as.integer(length(ts_data) * 0.7)
  idx2 <- as.integer(length(ts_data) * 0.2)
  
  ret$train <- ts(ts_data[1:idx], start = start((ts_data)), frequency = 8760)
  ret$test <- ts(ts_data[(idx+1):(idx+idx2)], start = index(ts_data)[idx+1], frequency = 8760)
  ret$val <- ts(ts_data[(idx+idx2+1):length(ts_data)], start = index(ts_data)[idx+idx2+1], frequency = 8760)
  
  ret$full <- ts_data
  
  return(ret)
}

hourlyTs <- SplitData(house_data)

plot(hourlyTs$train, xlim=c(1,9000))
lines(hourlyTs$test, col='red')
lines(hourlyTs$val, col='blue')

ExploreTs <- function(tsobj) {
  ts1 <- tsobj$full
  
  
  cat(sprintf(
    "Start: %s\nEnd: %s\nFrequency %s\n",
    paste(start(ts1), collapse = " "),
    paste(end(ts1), collapse = " "),
    frequency(ts1)
  ))
  
  cat("Plots a time series components, acf and pacf graphs")
  
  plotSubTitle = sprintf("%s - %s", tsobj$state, tsobj$dataset)
  
  print("Auto correlation plots")
  par(mfrow=c(1,2))
  ggacf(ts1)
  ggpacf(ts1)
}

ExploreTs(hourlyTs)


FittedModelDiagnostics <- function(tsfit) {
  # print(tsfit %>% summary())
  tsfit %>% checkresiduals() -> p
  
  print(p)
}

ForecastAndAccuracy <- function(tsobj, tsfit) {
  ret <- list()
  
  forecastHorizon = length(tsobj$test)
  ret$forecast <- forecast(tsfit, h = forecastHorizon)
  
  ret$accuracy <- accuracy(ret$forecast, tsobj$test)
  
  
  plotSubTitle = sprintf("%s - %s", tsobj$state, tsobj$dataset)
  autoplot(ret$forecast) +
    labs(y = "Demand (MWh)",
         subtitle = plotSubTitle) + 
    autolayer(tsobj$test, series = "Test")-> p
  print(p)
  
  
  ret$desc <- sprintf("Accuracy measures for %s %s STLF using %s method.", 
                      tsobj$state, 
                      tsobj$dataset, 
                      tsfit$method)
  
  print(ret$desc)
  print(ret$accuracy)
  
  return(ret)
}

ts1 <- hourlyTs$train

naiveFit1 <- naive(ts1, h=length(ts1))

FittedModelDiagnostics(naiveFit1)

autarimaFit1 <- auto.arima(ts1)
FittedModelDiagnostics(autarimaFit1)

logHourly <- hourlyTs
logHourly$train <- log(logHourly$train)
logHourly$test <- log(logHourly$test)
logHourly$full <- log(logHourly$full)

autoarimaFit2 <- auto.arima(logHourly$train)
FittedModelDiagnostics(autoarimaFit2)

tbatsFit <- tbats(ts1, use.box.cox = T,
                  use.damped.trend = NULL,
                  use.trend = NULL,
                  use.parallel = T)

FittedModelDiagnostics(tbatsFit)

stlfHourly <- stlf(ts1)
FittedModelDiagnostics(stlfHourly)

naiveRR <-ForecastAndAccuracy(hourlyTs,naiveFit1)
