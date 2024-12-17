import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Load the data from JSON file with error handling
file_path = '/Users/paullemaire/Documents/data2024.json'
try:
    df = pd.read_json(file_path, lines=True)
except ValueError as e:
    print(f"Error reading JSON file: {e}")
    exit(1)

user = 'Mu'

# Filter the messages written by "Alpha"
alpha_messages = df[df['Downlink-CRM-ClientID'] == user]

# Convert the 'Downlink-GTW-Timestamp' to datetime
alpha_messages['Downlink-GTW-Timestamp'] = pd.to_datetime(alpha_messages['Downlink-GTW-Timestamp'])

# Extract date and hour for grouping
alpha_messages['Date'] = alpha_messages['Downlink-GTW-Timestamp'].dt.date
alpha_messages['Hour'] = alpha_messages['Downlink-GTW-Timestamp'].dt.hour

# Group by date, hour, and device
grouped_messages = alpha_messages.groupby(['Date', 'Hour', 'Downlink-CRM-DeviceID']).size().reset_index(name='MessageCount')


# Get unique dates
unique_dates = grouped_messages['Date'].unique()

# Get unique devices
unique_devices = alpha_messages['Downlink-CRM-DeviceID'].unique()

# Create a color map
colors = cm.rainbow(np.linspace(0, 1, len(unique_devices)))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
fig.suptitle(f'Messages sent by {user}', fontsize=16)

for i, (ax, date) in enumerate(zip(axes.flatten(), unique_dates)):
    daily_data = grouped_messages[grouped_messages['Date'] == date]
    
    bottom = np.zeros(24)  # Initialize bottom array
    
    for i, (device, color) in enumerate(zip(unique_devices, colors)):  
        device_data = daily_data[daily_data['Downlink-CRM-DeviceID'] == device]
        message_counts = device_data.set_index('Hour')['MessageCount'].reindex(range(24), fill_value=0)
        ax.bar(range(24), message_counts, bottom=bottom, color=color, label=f'device{i}')
        bottom += message_counts  # Update bottom array

    ax.set_title(f'Date: {date}')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Message Count')
    ax.set_xticks(range(24))
    ax.set_ylim(bottom=0)  
# ax.legend(fontsize=2)  # Commented out to not show the legend

# Show the total number of devices
ax.text(0.5, 0.95, f'Total devices: {len(unique_devices)}', transform=ax.transAxes, ha='center')

# Rest of your code...

# If there are fewer dates than subplots, remove the unused subplots
if len(unique_dates) < len(axes.flatten()):
    for ax in axes.flatten()[len(unique_dates):]:
        ax.remove()

# Adjust layout
plt.tight_layout()
plt.show()