# Ex.No: 13 Learning – Use Supervised Learning  
### DATE: 28.10.2023
### REGISTER NUMBER : 212221040154
### AIM: 
To write a program to train the classifier for analyzing and predicting global average temperature over the next 25 years using the SARIMAX model, with a focus on data collection, preprocessing, and feature engineering to achieve accurate forecasting.
###  Algorithm:
* Collect and clean climate data.
* Create a time series with relevant features.
* Standardize data if needed.
* Split into training and testing sets.
* Train a SARIMAX model.
* Incorporate exogenous variables.
* Evaluate using metrics like MAE, MSE, and RMSE.
* Detect and handle outliers.
* Visualize predictions.
* Forecast the global temperature for 25 years.
* Present results and insights.
### Program:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("GlobalLandTemperatures_GlobalLandTemperaturesByCountry.csv")
print(data.head())
print(data.tail())
print(data.info())
print(data.nunique())
print((data.isnull().sum()/(len(data)))*100)
data['dt'] = pd.to_datetime(data['dt'])
data['year'] = data['dt'].dt.year
data['month'] = data['dt'].dt.month
bins = [1700, 1800, 1900, 2000, 2014]
labels = ['18th century', '19th century', '20th century', '21st century']
data['year_category'] = pd.cut(data['year'], bins=bins, labels=labels, right=False)
columns_to_scale = ['AverageTemperature', 'AverageTemperatureUncertainty']
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data[columns_to_scale]), columns=columns_to_scale)
data_encoded = pd.concat([data[['dt', 'Country', 'year', 'month', 'year_category']], data_scaled], axis=1)
data_encoded.to_csv("scaled_data.csv", index=False)
import matplotlib.pyplot as plt
import seaborn as sns
#Load the dataset
data = pd.read_csv("/content/scaled_data.csv")

# Convert the "dt" column to datetime format
data['dt'] = pd.to_datetime(data['dt'])

# Extract the decade from the "dt" column
data['decade'] = data['dt'].dt.year // 10 * 10

# Calculate the average temperature per country and decade
avg_temp = data.groupby(['decade', 'Country'])['AverageTemperature'].mean().reset_index()

# Find the country with the lowest average temperature for each decade
min_temp_per_decade = avg_temp.groupby('decade')['AverageTemperature'].idxmin()
countries_lowest_temp_decade = avg_temp.loc[min_temp_per_decade, ['decade', 'Country', 'AverageTemperature']]

# Visualize the countries with the lowest average temperature for each decade
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=countries_lowest_temp_decade, x='decade', y='AverageTemperature', hue='Country')
plt.title("Country with Lowest Average Temperature (in Celsius) for Each Decade")
plt.xlabel("Decade (Duration: 10 years)")
plt.ylabel("Average Temperature (Celsius)")
plt.xticks(rotation=45)
plt.show()
# Find the country with the highest average temperature for each decade
max_temp_per_decade = avg_temp.groupby('decade')['AverageTemperature'].idxmax()
countries_highest_temp = avg_temp.loc[max_temp_per_decade, ['decade', 'Country', 'AverageTemperature']]

# Visualize the countries with the highest average temperature for each decade
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=countries_highest_temp, x='decade', y='AverageTemperature', hue='Country')
plt.title("Country with Highest Average Temperature (in Celsius) for Each Decade")
plt.xlabel("Decade (Duration: 10 years)")
plt.ylabel("Average Temperature (Celsius)")
plt.xticks(rotation=45)
plt.show()
#Extract the century from the "dt" column
data['century'] = data['dt'].dt.year // 100

# Calculate the average temperature per country and century
avg_temp = data.groupby(['century', 'Country'])['AverageTemperature'].mean().reset_index()

# Find the country with the lowest average temperature for each century
min_temp_per_century = avg_temp.groupby('century')['AverageTemperature'].idxmin()
countries_lowest_temp_century = avg_temp.loc[min_temp_per_century, ['century', 'Country', 'AverageTemperature']]

# Visualize the countries with the lowest average temperature for each century
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=countries_lowest_temp_century, x='century', y='AverageTemperature', hue='Country')
plt.title("Country with Lowest Average Temperature (in Celsius) for Each Century")
plt.xlabel("Century (Duration: 100 years)")
plt.ylabel("Average Temperature (Celsius)")
plt.xticks(rotation=45)
plt.show()

# Find the country with the highest average temperature for each century
max_temp_per_century = avg_temp.groupby('century')['AverageTemperature'].idxmax()
countries_highest_temp = avg_temp.loc[max_temp_per_century, ['century', 'Country', 'AverageTemperature']]

# Print the countries with the highest average temperature for each century
print(countries_highest_temp)

# Visualize the countries with the highest average temperature for each century
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=countries_highest_temp, x='century', y='AverageTemperature', hue='Country')
plt.title("Country with Highest Average Temperature (in Celsius) for Each Century")
plt.xlabel("Century (Duration: 100 years)")
plt.ylabel("Average Temperature (Celsius)")
plt.xticks(rotation=45)
plt.show()

# Calculate the average temperature uncertainty over time
temp_uncertainty = data.groupby(data['dt'].dt.year)['AverageTemperatureUncertainty'].mean()

# Plot the temperature uncertainty over time
plt.figure(figsize=(10, 6))
plt.plot(temp_uncertainty.index, temp_uncertainty.values)
plt.title("Temperature Uncertainty Over Time")
plt.xlabel("Year")
plt.ylabel("Temperature Uncertainty")
plt.show()

# Calculate the temperature variation (standard deviation) over time
temp_variation = data.groupby(data['dt'].dt.year)['AverageTemperature'].std()

# Create a scatter plot for temperature variation over time
plt.figure(figsize=(10, 6))
plt.scatter(temp_variation.index, temp_variation.values)
plt.title("Temperature Variation Over Time")
plt.xlabel("Year")
plt.ylabel("Temperature Variation (Celsius)")
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the encoded dataset
data_encoded = pd.read_csv("/content/scaled_data.csv")

# Prepare the data for time series forecasting
data_encoded['dt'] = pd.to_datetime(data_encoded['dt'])
data_encoded = data_encoded.set_index('dt')
data_encoded = data_encoded.resample('Y').mean()  # Resample data to yearly average
data_encoded = data_encoded['1743':'2013']  # Filter data up to 2013

# Drop rows with missing values
data_encoded.dropna(inplace=True)

# Train the SARIMA model
model = SARIMAX(data_encoded['AverageTemperature'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Make predictions for the historical data
historical_predictions = model_fit.predict(start=data_encoded.index[0], end=data_encoded.index[-1])

# Calculate the root mean squared error (RMSE)
rmse = mean_squared_error(data_encoded['AverageTemperature'], historical_predictions, squared=False)

# Make predictions for the next 25 years
future_dates = pd.date_range(start='2014-01-01', periods=25, freq='Y')
forecast = model_fit.predict(start=len(data_encoded), end=len(data_encoded) + 24)

# Plot the historical data and forecast
plt.figure(figsize=(12, 6))
plt.plot(data_encoded.index, data_encoded['AverageTemperature'], label='Historical Data')
plt.plot(future_dates, forecast, label='Forecast')
plt.xlabel('Year')
plt.ylabel('Average Temperature (in Celsius)')
plt.title('Average Temperature Forecast for the Next 25 Years')
plt.legend()
plt.show()

print(f"RMSE: {rmse}")
```
### Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 577462 entries, 0 to 577461
Data columns (total 4 columns):
 #   Column                         Non-Null Count   Dtype  
---  ------                         --------------   -----  
 0   dt                             577462 non-null  object 
 1   AverageTemperature             544811 non-null  float64
 2   AverageTemperatureUncertainty  545550 non-null  float64
 3   Country                        577462 non-null  object 
dtypes: float64(2), object(2)
memory usage: 17.6+ MB
None
```
```
dt                               0.000000
AverageTemperature               5.654225
AverageTemperatureUncertainty    5.526251
Country                          0.000000
dtype: float64
```
![image](https://github.com/Siddarthan999/AI_Lab_2023-24/assets/91734840/bbf04f9d-6ef7-4f71-b84e-329c2a083bdb)

![image](https://github.com/Siddarthan999/AI_Lab_2023-24/assets/91734840/91039519-c839-4025-957f-12484c7e2268)

```
     century    Country  AverageTemperature
64        17  Sri Lanka            0.889449
135       18   Djibouti            1.033596
374       19   Djibouti            1.055925
617       20   Djibouti            1.148273
```
![image](https://github.com/Siddarthan999/AI_Lab_2023-24/assets/91734840/9ec2052f-0a20-467d-9e9d-1fec94ca6bfc)

![image](https://github.com/Siddarthan999/AI_Lab_2023-24/assets/91734840/16c7bcdd-7656-4b0f-9a0a-4ca6a1a3de76)

![image](https://github.com/Siddarthan999/AI_Lab_2023-24/assets/91734840/f9c42d27-6941-4ee9-8e3d-c150a8d98b9c)

![image](https://github.com/Siddarthan999/AI_Lab_2023-24/assets/91734840/b9237171-92eb-4b63-af5e-62cda1ab6e4a)
```
RMSE: 0.14366912040368834
```
### Result:
Thus, the system was trained successfully, and the prediction was carried out.
