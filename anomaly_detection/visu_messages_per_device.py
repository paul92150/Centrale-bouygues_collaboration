import pandas as pd
import matplotlib.pyplot as plt

# Load the data from JSON file with error handling
file_path = '/Users/paullemaire/Documents/data2024.json'
try:
    df = pd.read_json(file_path, lines=True)
except ValueError as e:
    print(f"Error reading JSON file: {e}")
    exit(1)

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Extract date, hour, and user for aggregation
df['date'] = df['Downlink-GTW-Timestamp'].dt.date
df['hour'] = df['Downlink-GTW-Timestamp'].dt.hour

# Filter data for user 'Alpha'
df_alpha = df[df['Downlink-CRM-ClientID'] == 'Alpha']

# Debug: Check the shape and head of the filtered dataframe
print("Filtered DataFrame shape:", df_alpha.shape)
print(df_alpha.head())

# Group by date, hour, and device ID and count occurrences
count_per_hour_date_device_alpha = df_alpha.groupby(['date', 'hour', 'Downlink-CRM-DeviceID']).size().unstack(fill_value=0)

# Debug: Print the head of the grouped data
print("Grouped Data:")
print(count_per_hour_date_device_alpha.head(24))

# Specify the three known dates
dates = [pd.to_datetime('2024-01-27').date(), pd.to_datetime('2024-01-28').date(), pd.to_datetime('2024-01-29').date()]

# Create a figure with 2 columns and 2 rows of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 8))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Generate a custom list of colors
unique_devices = count_per_hour_date_device_alpha.columns.unique()
num_colors = len(unique_devices)
colors = plt.get_cmap('tab20', num_colors).colors

# Create a dictionary to map each device to a unique color
color_mapping = {device: colors[i] for i, device in enumerate(unique_devices)}

# Plot each date's data in its respective subplot
for i, date in enumerate(dates):
    if date in count_per_hour_date_device_alpha.index.get_level_values('date'):
        data_for_date = count_per_hour_date_device_alpha.loc[date]
        
        # Debug: Print the data for the current date
        print(f"Data for {date}:")
        print(data_for_date)
        
        # Plot bars for each device
        for k, device in enumerate(data_for_date.columns):
            axes[i].bar(data_for_date.index, data_for_date[device], label=f'device{k}', color=color_mapping[device])
        
        axes[i].set_title(f'Total messages for Alpha on {date}')
        axes[i].set_xlabel('Hour of the Day')
        axes[i].set_ylabel('Total Messages')
        axes[i].set_xticks(range(24))
        axes[i].set_xticklabels(range(24), rotation=0)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend(title='Device')
    else:
        axes[i].set_title(f'No data available for {date}')
        axes[i].set_xticks(range(24))
        axes[i].set_xticklabels(range(24), rotation=0)

# Turn off the last subplot (bottom-right) if it's not used
if len(dates) < len(axes):
    axes[-1].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()
