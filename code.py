import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load historical price data for each metal commodity
# Replace 'data.csv' with the actual file containing your data
data = pd.read_csv('data.csv')

# Assuming your data has columns: 'Date', 'Copper', 'Steel', 'Oil', 'Rubber', 'Alloys'
# You may need to preprocess your data, e.g., converting 'Date' column to datetime format

# Perform price prediction for Copper using Linear Regression
copper_data = data[['Date', 'Copper']].dropna()  # Extracting Copper data and removing missing values
X = np.array(copper_data.index).reshape(-1, 1)  # Using index as feature (assuming data is sequential)
y = np.array(copper_data['Copper'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for Copper:", mse)

# Plot actual vs predicted prices for Copper
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Copper Price Prediction')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()

# Similarly, you can perform price prediction for other metal commodities
# You can also implement trend analysis using technical indicators and chart patterns
# Additionally, consider incorporating sentiment analysis for market sentiment assessment

#upon running in Notebook accuracy is 40% without any further alteration, and 57% with GB. 
