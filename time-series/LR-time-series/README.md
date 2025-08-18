## Linear Regression With Time Series

This folder contains my solution to the **"Linear Regression With Time Series"** exercise from the [Kaggle Time Series Course](https://www.kaggle.com/learn/time-series).


During training, the regression algorithm learns values for the parameters weight_1, weight_2, and bias that best fit the target

This algorithm is often called ordinary least squares since it chooses values that minimize the squared error between the target and the predictions

The weights are also called regression coefficients and the bias is also called the intercept because it tells you where the graph of this function crosses the y-axis

There are two kinds of features unique to time series: time-step features and lag features

Time-step features are features we can derive directly from the time index. The most basic time-step feature is the time dummy, which counts off time steps in the series from beginning to end

A series is time dependent if its values can be predicted from the time they occured. 

To make a lag feature we shift the observations of the target series so that they appear to have occured later in time