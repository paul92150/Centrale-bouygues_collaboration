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

# Extract date, minute, and user for aggregation
df['date'] = df['Downlink-GTW-Timestamp'].dt.date
df['minute'] = df['Downlink-GTW-Timestamp'].dt.floor('T')  # Floor to the nearest minute

# Filter for JOIN_ACCEPT
df_join_accept = df[df['Downlink-LORA-MType'] == 'JOIN_ACCEPT']

# Group by date, minute, and user and count occurrences
count_per_minute_date_user = df_join_accept.groupby(['date', 'minute', 'Downlink-CRM-ClientID']).size().unstack(fill_value=0)

# Specify the date to plot
plot_date = pd.to_datetime('2024-01-29').date()

# Check if the date is in the data
if plot_date in count_per_minute_date_user.index.levels[0]:
    # Filter data for the specific date
    data_for_date = count_per_minute_date_user.loc[plot_date]
    
    # Create a figure with 1 column and 1 row of subplots (can adjust if plotting more dates)
    fig, ax = plt.subplots(figsize=(18, 8))

    # Define a color map
    colors = plt.get_cmap('tab20', len(data_for_date.columns))

    # Plot each user's data in the subplot
    for user in data_for_date.columns:
        ax.bar(data_for_date.index, data_for_date[user], label=user, color=colors(data_for_date.columns.get_loc(user)))
    
    ax.set_title('Count of join requests on 2024-01-29')
    ax.set_xlabel('Time of the Day')
    ax.set_ylabel('Count')
    ax.xaxis_date()  # Ensure x-axis is formatted as dates
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='User')

    # Adjust layout
    plt.tight_layout()
    plt.show()
else:
    print(f"No data available for {plot_date}")

