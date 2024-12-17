import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from JSON file
file_path = '/Users/paullemaire/Documents/data2024.json'
df = pd.read_json(file_path, lines=True)

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Extract date and hour for aggregation
df['date'] = df['Downlink-GTW-Timestamp'].dt.date
df['hour'] = df['Downlink-GTW-Timestamp'].dt.hour

# Aggregate message counts by date and hour
message_counts = df.groupby(['date', 'hour']).size().reset_index(name='message_count')

# Debugging: Print the first few rows of the message counts dataframe
print("First few rows of the message counts dataframe:\n", message_counts.head())

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Line Plot for Message Frequency Over Time
# Aggregate messages by date for line plot
daily_messages = df.groupby('date').size().reset_index(name='message_count')

# Debugging: Print the aggregated message counts
print("Aggregated message counts:\n", daily_messages)

# Create a line plot for message frequency over time
axes[0].plot(daily_messages['date'], daily_messages['message_count'], marker='o')
axes[0].set_title('Message Frequency Over Time')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Message Count')
axes[0].tick_params(axis='x', rotation=45)

# Heatmap of Message Frequency
# Pivot data for heatmap
heatmap_data = message_counts.pivot(index='date', columns='hour', values='message_count')

# Create a heatmap
sns.heatmap(heatmap_data, cmap='viridis', cbar=True, annot=True, fmt='g', ax=axes[1])
axes[1].set_title('Heatmap of Message Frequency by Date and Hour')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Date')

plt.tight_layout()
plt.show()
