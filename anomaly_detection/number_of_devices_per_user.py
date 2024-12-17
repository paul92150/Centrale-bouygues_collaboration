import pandas as pd

# load the data from JSON file with error handling
file_path = '/Users/paullemaire/Documents/data2024.json'
try:
    df = pd.read_json(file_path, lines=True)
except ValueError as e:
    print(f"Error reading JSON file: {e}")
    exit(1)

# group by user and device, then count unique devices per user
devices_per_user = df.groupby('Downlink-CRM-ClientID')['Downlink-CRM-DeviceID'].nunique()

# convert the result to a dictionary
devices_per_user_dict = devices_per_user.to_dict()

print(devices_per_user_dict)
#print the total number of devices
print(f"Total number of devices: {sum(devices_per_user_dict.values())}")


