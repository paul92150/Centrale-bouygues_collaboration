import pandas as pd
import matplotlib.pyplot as plt

# Load the data from JSON file
file_path = '/Users/paullemaire/Documents/data2024.json'
df = pd.read_json(file_path, lines=True)

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Sort the dataframe by timestamp
df = df.sort_values(by='Downlink-GTW-Timestamp')

# Calculate the time difference between consecutive messages
df['time_diff'] = df['Downlink-GTW-Timestamp'].diff().dropna()

# Drop the first row as it will have NaN value in 'time_diff'
df = df.dropna(subset=['time_diff'])

# Convert time difference to seconds
df['time_diff_seconds'] = df['time_diff'].dt.total_seconds()

# Debugging: Print the first few rows of the dataframe
print("First few rows of the dataframe with time differences:\n", df.tail())

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(df['Downlink-GTW-Timestamp'], df['time_diff_seconds'], marker='', linestyle='-', markersize=2)
plt.xlabel('Time')
plt.ylabel('Time Between Messages (seconds)')
plt.title('Time Between Consecutive Messages')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
