import pandas as pd

# Load the data from JSON file with error handling
file_path = '/Users/paullemaire/Documents/data2024.json'
try:
    df = pd.read_json(file_path, lines=True)
except ValueError as e:
    print(f"Error reading JSON file: {e}")
    exit(1)

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Extract date, minute, and user for aggregation
df['date'] = df['Downlink-GTW-Timestamp'].dt.date

df_before_date = df[df['date'].isin([pd.to_datetime('2024-01-27').date(), pd.to_datetime('2024-01-28').date()])]

device_message_frequency = df_before_date.groupby('Downlink-CRM-DeviceID').size().reset_index(name='MessageCount')

# Divide the 'MessageCount' column by 48
device_message_frequency['MessageCount'] = device_message_frequency['MessageCount'] / 48

# Calculate the average message frequency
average_message_frequency = device_message_frequency['MessageCount'].mean()

print(average_message_frequency)
print(device_message_frequency.head())