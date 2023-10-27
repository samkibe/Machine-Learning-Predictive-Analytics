#!/usr/bin/env python
# coding: utf-8

# - Apply RNN to perform predictive analytics.
# - Predictive analytics on Kenya's - Consumer Price Indices (CPI) 
# - Considering the data goes back to 2019 we would need to investigate and find out if models could have predicted the current prices using the already experienced past prices.
# - Dataset from HDX: (https://data.humdata.org/dataset/86263012-414d-434e-b072-cd1383f8291d)
# - Therefore we would need to perform a time series forecasting to predict the current consumer price index using consumer price index from 2019
# #pip install pandas numpy matplotlib statsmodels

# In[1]:


#suppress warnings for a clean notebook just to moderate error messages
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Loading historical Kenya CPI data 
# I had to clean the date column because the entries were wrongly set using microsoft excel
df = pd.read_csv('#') #download data from url provided
df['Date'] = pd.to_datetime(df['Date']) 
df.set_index('Date', inplace=True)
data = df['Kenya CPI'].values.reshape(-1, 1) 



# - CPI utilizes weighted averages, in our case a basket has been randomly  selected 
# - A basket of goods and services that are commonly purchased by a typical household.
# - Then these products and services commonly purchased by a typical household are averaged using varying levels of importance or relevance in the typical consumer's spending patterns.
# - Each item in the basket is now assigned a weight that reflects its relative importance in the average consumer's budge.

# In[3]:


#Data exploration
df.head()
# results with weighted averages


# In[6]:


#Data exploration

# Visualize the data
#plt.figure(figsize=(12, 6))
plt.plot(df)
plt.title('Historical Kenya Consumer Price Index')
plt.xlabel('Year')
plt.ylabel('Kenya Consumer Price Index')
plt.show()


# In[7]:


correlation=df.corr()
correlation['Kenya CPI'].sort_values(ascending=True)


# - Basically since we want to predict the consumer price index over time, we would want to use dates as the dependent variable 
# - And the overall Kenya CPI as the independent variable which would use the dependent variable for generalization

# In[8]:


#Data exploration

# Extract relevant columns of interest
#independent
df = df[['Kenya CPI']]
#depedent if and only if
#df = df[['Date']]
df.head()


# In[9]:


#look for missing values in our select column of interest
df.isna().sum()
#the first instance, it found nan values but after cleaning the data in excel the algorithmn run zero nan values


# In[10]:


plt.plot(data)
plt.title('Historical Price Index')
plt.xlabel('Year')
plt.ylabel('Kenya Consumer Price Index')
plt.show()


# In[12]:


# Normalize the data using Min-Max scaling
# Min-Max scaling is a technique for normalizing the data, 
# ........which means transforming it in a way that the values fall within a specific range
# Scaling ensures the data has a similar scale across features and that it falls within a range that is 
# .......... suitable for the neural network to learn effectively.
scaler = MinMaxScaler()
data = scaler.fit_transform(data) # data defined above


# In[13]:


# standard basic RNN procedures

# EG Create sequences for training
# Sequences are essential for training RNNs because they can capture temporal dependencies in the data
sequence_length = 10  # You can adjust this value
X, y = [], []

# for loop ensures that you don't go out of bounds when creating sequences. 
for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)


# In[14]:


# Split the data into training and testing sets
# Recognize we are using 80% for training and 20 for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# In[15]:


# Build the Long Short-Term Memory (LSTM) model
# example of a simple RNN model using the Long Short-Term Memory (LSTM) architecture
model = keras.Sequential([
    keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    keras.layers.Dense(1)
])
#We can also try SGD or RMSprop in place of adam or vise versa. 
#Basically these concepts adapts the learning rate during training to improve convergence
model.compile(optimizer='adam', loss='mean_squared_error')


# In[17]:


# Train the model
# epoch def: This parameter specifies the number of times the model will iterate through our entire dataset. 
#therefore an epoch represents one complete pass through the entire training dataset
model.fit(X_train, y_train, epochs=50, batch_size=30)

#if adam was used loss started at 0.0042 - 0.3880- 0.450 
#if sgd was used insted of adam - loss started at 0.3907 - 0.4032
# if RMSprop was used - loss started at 0.4250


# - A lower loss for our case indicates that the model's predictions are getting closer to the actual target values.
# - Which can only mean the model is learning and improving its ability to make predictions on your training data which is wuite incredible and satisfying
# - The firt training i did gave me a loss of an appoximate 36000.00 loss value which was some very bad results

# In[18]:


print("Actual values in the array:")
print(X_test)
#print("Actual value for the first sample:", X_test[0])


# In[19]:


# Make predictions on the test data
predictions = model.predict(X_test)
# We now need to Inverse transform the predictions to our original scale, 
#.....because we used scaler as a standard procedure for RNNS
# ... because now we want to interpret the predictions in their original context.
predictions = scaler.inverse_transform(predictions)  # Inverse scaling
predictions

pred_y = model.predict(X_test)
pred_y #both give same predictions


# In[20]:


#I took an approximate average of the actual target values for the test sample
#actual_value = 1.0  
actual_value = 0.6
# The model's prediction as above
prediction = 0.71092254  
# Calculate metrics
mse = mean_squared_error([actual_value], [prediction])
rmse = np.sqrt(mse)
mae = mean_absolute_error([actual_value], [prediction])
r2 = r2_score([actual_value], [prediction])

print(f'Mean Squared Error (MSE): {mse}') #Lower values indicate better performance.
print(f'Root Mean Squared Error (RMSE): {rmse}') #average magnitude of the error, the lower the better
print(f'Mean Absolute Error (MAE): {mae}') 
print(f'R-squared (R²): {r2}')



# Our values for MSE, RMSE, and MAE suggest that the predictions are very close to the true target values, and there may not be much variation in the true target values, which now leads to the "nan" R² value.

# In[21]:


# Evaluate the model
loss = model.evaluate(X_test, y_test)
#loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')


# In summarry
# - The loss value above represents how well our model is performing on the test data, and since they are Low this indicates a better performance. 
# - Basically it has measured how close the model's predictions are to the true target values. 

# In[22]:


#import matplotlib.pyplot as plt
# Visualize predictions
#plt.figure(figsize=(12, 6))
#plt.plot(df.index[-len(y_test):], y_test, label='Original Values')
#plt.plot(df.index[-len(y_test):], predictions, label='Predictions', linestyle='dashed')
#plt.xlabel('YEAR')
#plt.ylabel('Kenya CPI')
#plt.legend()
#plt.title('Price Index Prediction with LSTM')
#plt.show()



# - Let us now try to predict or forecaste future values of the Kenya Consumer Price Index (CPI). 
# - Basically, probably of close present using these experienses

# In[23]:


# number of future time periods to forecast defined here
future_periods = 30  # wse can always adjust this as needed

# Create a date range for the future time periods
future_dates = pd.date_range(start=df.index[-1], periods=future_periods + 1, closed='right')

# Initialize an array for forecasted values
forecasted_values = []

# Make predictions for future time periods
for i in range(future_periods):
    
    # Select the most recent sequence of data (this is defined on cell one for kenya cpi)
    recent_sequence = data[-sequence_length:]

    # Reshape the sequence to match the model's input shape
    recent_sequence = recent_sequence.reshape(1, sequence_length, 1)

    # Predict the next value
    next_value = model.predict(recent_sequence)

    # Append the predicted value to the dataset
    data = np.append(data, next_value, axis=0) #(data:this is defined on cell one for kenya cpi)

    # Append the predicted value to the list of forecasted values
    forecasted_values.append(next_value)

# Inverse transform the forecasted values to the original scale
#forecasted_values = scaler.inverse_transform(forecasted_values).flatten()
predictions = scaler.inverse_transform(predictions)

# Create a DataFrame for the forecasted values
forecasted_df = pd.DataFrame({'Date': future_dates[0:], 'Forecasted_CPI': forecasted_values})

# Concatenate the original data with the forecasted values
complete_df = pd.concat([df, forecasted_df], axis=0)
# Visualize the original data and the forecasted values
plt.figure(figsize=(12, 6))
plt.plot(complete_df['Date'], complete_df['Kenya CPI'], label='Original Data')
plt.plot(forecasted_df['Date'], forecasted_df['Forecasted_CPI'], label='Forecasted Values', linestyle='dashed')
plt.title('Forecasting Future Kenya CPI with LSTM')
plt.xlabel('Year')
plt.ylabel('Kenya Consumer Price Index')
plt.legend()
plt.show()


# In[ ]:




