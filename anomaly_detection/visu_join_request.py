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

# Filter for JOIN_ACCEPT
df_join_accept = df[df['Downlink-LORA-MType'] == 'JOIN_ACCEPT']

# Group by date, hour, and user and count occurrences
count_per_hour_date_user = df_join_accept.groupby(['date', 'hour', 'Downlink-CRM-ClientID']).size().unstack(fill_value=0)

# Specify the three known dates
dates = [pd.to_datetime('2024-01-27').date(), pd.to_datetime('2024-01-28').date(), pd.to_datetime('2024-01-29').date()]

# Create a figure with 2 columns and 2 rows of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 8))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Define a color map
colors = plt.get_cmap('tab20', len(df['Downlink-CRM-ClientID'].unique()))

# Plot each date's data in its respective subplot
for i, date in enumerate(dates):
    if date in count_per_hour_date_user.index:
        data_for_date = count_per_hour_date_user.loc[date]
        
        for user in data_for_date.columns:
            axes[i].bar(data_for_date.index, data_for_date[user], label=user, color=colors(data_for_date.columns.get_loc(user)))

        axes[i].set_title(f'Count of join request on {date}')
        axes[i].set_xlabel('Hour of the Day')
        axes[i].set_ylabel('Count')
        axes[i].set_xticks(range(24))
        axes[i].set_xticklabels(range(24), rotation=0)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend(title='User')
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

