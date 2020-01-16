# R5dqiWFpt-KFzVXsJXJ4
# quandl for financial data
import quandl
# pandas for data manipulation
import pandas as pd, matplotlib.pyplot as plt

quandl.ApiConfig.api_key = 'R5dqiWFpt-KFzVXsJXJ4'# Retrieve TSLA data from Quandl

reliance = quandl.get('NSE/RELIANCE')
reliance.head(5)
#type(reliance)
reliance.columns
reliance2 = reliance

#Selecting data between 2004 & 2019
reliance = reliance2.iloc[1454:5183]


# The adjusted close accounts for stock splits, so that is what we should graph
plt.plot(reliance.index, reliance['Close'])
plt.title('Reliance Stock Price')
plt.xlabel('Year')
plt.ylabel('Price (Rs)');

plt.show()

reliance_shares = {2019: 6.33e9, 2018: 6.33e9, 2017: 2.95e9, 2016: 2.94e9, 2015: 2.94e9, 2014: 2.94e9, 2013: 2.94e9, 2012: 2.98e9, 2011: 2.98e9, 2010: 2.98e9, 2009: 2.75e9, 2008: 2.91e9, 2007: 2.79e9, 2006: 2.79e9, 2005: 2.79e9, 2004: 2.79e9}

# Create a year column 
reliance['Year'] = reliance.index.year

# Take Dates from index and move to Date column 
reliance.reset_index(level=0, inplace = True)
reliance['cap'] = 0

# Calculate market cap for all years
for i, year in enumerate(reliance['Year']):
    # Retrieve the shares for the year
    shares = reliance_shares.get(year)
    
    # Update the cap column to shares times the price
    reliance.ix[i, 'cap'] = shares * reliance.ix[i, 'Close']
    
# Select only the relevant columns
market_cap = reliance[['Date', 'cap']]

# Divide to get market cap in billions of dollars
market_cap.head(5)

plt.figure(figsize=(10, 8))
plt.plot(market_cap['Date'], market_cap['cap'], 'b-', label = 'Reliance')
plt.xlabel('Date'); plt.ylabel('Market Cap Rupees (Rs)'); plt.title('Market Cap of Reliance')
plt.legend();

import numpy as np

# The time when Reliance was valued the highest till now
highest_date = market_cap.loc[market_cap['cap'].idxmax(), 'Date']
print("Reliance was valued the highest in {} .".format(highest_date.date()))

import fbprophet
# Prophet requires columns ds (Date) and y (value)
reliance = reliance.rename(columns={'Date': 'ds', 'cap': 'y'})# Put market cap in Rupees
reliance['y'] = reliance['y'] / 1e9# Make the prophet model and fit on the data
reliance_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15)
reliance_prophet.fit(reliance)

# Make a future dataframe for 2 years
reliance_forecast = reliance_prophet.make_future_dataframe(periods=365 * 2, freq='D')# Make predictions
reliance_forecast = reliance_prophet.predict(reliance_forecast)

reliance_prophet.plot(reliance_forecast, xlabel = 'Date', ylabel = 'Market Cap (Rupees)')
plt.title('Market Cap of Reliance');

# Plot the trends and patterns
reliance_prophet.plot_components(reliance_forecast)