import pandas as pd
import matplotlib.pyplot as plt

# Load the data from JSON file
file_path = '/Users/paullemaire/Documents/data2024.json'
df = pd.read_json(file_path, lines=True)

#filter the JOIN_ACCEPT messages
df = df[df["Downlink-LORA-MType"] == 'JOIN_ACCEPT'] 
# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Extract date and minute for aggregation
df['date_minute'] = df['Downlink-GTW-Timestamp'].dt.floor('T')

# Aggregate message counts by date and minute
message_counts = df.groupby('date_minute').size().reset_index(name='message_count')

# Debugging: Print the first few rows of the message counts dataframe
print("First few rows of the message counts dataframe:\n", message_counts.head())

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(message_counts['date_minute'], message_counts['message_count'], marker='', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Message Count (Log Scale)')
plt.title('Total Number of Messages Per Minute')
plt.yscale('log')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
