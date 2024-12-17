import pandas as pd
import matplotlib.pyplot as plt

number_of_devices_per_user = {'Alpha': 50, 'Beta': 76, 'Delta': 51, 'Epsilon': 14, 'Eta': 10, 'Evil': 5, 'Gamma': 132, 'Iota': 14, 'Kappa': 14, 'Lambda': 11, 'Mu': 9, 'Nu': 15, 'Theta': 8, 'Zeta': 18}

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

# Group by date, hour, and user and count occurrences
count_per_hour_date_user = df.groupby(['date', 'hour', 'Downlink-CRM-ClientID']).size().unstack(fill_value=0)

# Calculate messages per device
for user in count_per_hour_date_user.columns:
    if user in number_of_devices_per_user:
        count_per_hour_date_user[user] = count_per_hour_date_user[user] / number_of_devices_per_user[user]

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

        axes[i].set_title(f'Messages per device on {date}')
        axes[i].set_xlabel('Hour of the Day')
        axes[i].set_ylabel('Messages per Device')
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
