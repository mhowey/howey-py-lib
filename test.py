import pandas as pd
import matplotlib.pyplot as plt
print("Testing Pandas...")

csv = pd.read_csv('weather_data.csv')

print(pd.DataFrame.describe(csv))

plt.scatter(csv['Temperature_C'], csv['Humidity_pct'])
plt.xlabel('Temperature C')
plt.ylabel('Humidity %')
plt.title('Temperature vs Humidity')
plt.show()

