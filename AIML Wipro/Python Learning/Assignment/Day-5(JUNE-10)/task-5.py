# Create Basic Visualizations to Explore Data
# Load a time series dataset.
# Create a line plot to visualize the trend over time.
# Display the plot.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
dates = pd.date_range('2024-06-10', periods=10)
values = np.random.randn(10).cumsum()

date_series_data = pd.DataFrame({'Date': dates, 'Value': values})

plt.figure(figsize=(13, 5))
plt.plot(date_series_data['Date'], date_series_data['Value'], marker='o', linestyle='-', color='b')
plt.title('Date')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()
