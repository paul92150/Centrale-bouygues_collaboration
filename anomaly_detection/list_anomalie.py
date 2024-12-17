import pandas as pd

#anomalies to detect
n = 3

# Load the data from JSON file with error handling
file_path = '/Users/paullemaire/Documents/data2024.json'
try:
    df = pd.read_json(file_path, lines=True)
except ValueError as e:
    print(f"Error reading JSON file: {e}")
    exit(1)

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Extract date for filtering
df['date'] = df['Downlink-GTW-Timestamp'].dt.date

# Define the dates of interest
dates_of_interest = [pd.to_datetime('2024-01-26').date(), pd.to_datetime('2024-01-27').date()]

# Get the unique device IDs
device_ids = df['Downlink-CRM-DeviceID'].unique()

# List to store instances where time between responses is 3 times the mean time
anomalies = []

# Iterate over each device
for device_id in device_ids:
    # Filter rows for the specified device ID
    df_device = df[df['Downlink-CRM-DeviceID'] == device_id]
    
    # Check if the DataFrame is empty before proceeding
    if df_device.empty:
        continue

    # Filter for the dates of interest
    df_device = df_device[df_device['date'].isin(dates_of_interest)]
    
    # Sort by timestamp
    df_device = df_device.sort_values('Downlink-GTW-Timestamp')
    
    # Calculate time difference between consecutive responses
    df_device['time_diff'] = df_device['Downlink-GTW-Timestamp'].diff().dt.total_seconds()
    
    # Drop the first row with NaN time difference
    df_device = df_device.dropna(subset=['time_diff'])

    # Calculate the average time difference for the specified dates
    avg_time_diff = df_device['time_diff'].mean()
    
    # Identify instances where time difference is 3 times the mean time
    anomalies.extend(df_device[df_device['time_diff'] > n * avg_time_diff].to_dict('records'))

# Print the list of anomalies
print(f"Anomalies where time between responses is {n} times the mean time:")
for anomaly in anomalies:
    print(anomaly)
print(len(anomalies))
