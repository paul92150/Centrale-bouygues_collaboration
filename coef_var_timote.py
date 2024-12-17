import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the data from JSON file with error handling
file_path = '/Users/paullemaire/Documents/data2024.json'
try:
    df = pd.read_json(file_path, lines=True)
except ValueError as e:
    print(f"Error reading JSON file: {e}")
    exit(1)

# removing the last part where the DDoS happens
df = df[df['Downlink-GTW-Timestamp'] <= '2024-01-29 00:00:00']

# adding the difference variable
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])
df = df.sort_values(['Downlink-CRM-DeviceID', 'Downlink-GTW-Timestamp'])
df['diff_seconds'] = df.groupby('Downlink-CRM-DeviceID')['Downlink-GTW-Timestamp'].diff().dt.total_seconds()

final_data = {}
aux_data = []
factor = 2
for i in df['Downlink-CRM-DeviceID'].unique():
    device_data = df[df['Downlink-CRM-DeviceID'] == i]['diff_seconds']
    value = round(device_data.std()/device_data.mean(),2)
    if not value in final_data.keys():
        final_data[value] = 0
    final_data[value] += 1
    aux_data.append((i, value))

final_data = dict(sorted(final_data.items(), key=lambda x: x[0]))
aux_data = list(sorted(aux_data, key=lambda x: x[1]))[::-1]
print([aux_data[i] for i in range(len(aux_data)) if aux_data[i][1] <= 0.01])
print(aux_data)
# pretty colors
palette = sns.color_palette('flare', n_colors=len(final_data.values())) 

plt.bar(np.array(list(final_data.keys()))*100, np.array(list(final_data.values())), width=1, color=palette)
plt.gca().yaxis.grid(True)
#plt.legend()
plt.title("number of devices with a given coefficient of variation")
plt.xlabel("Coefficient of variation")
plt.ylabel("number of devices")
plt.show()