import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Define the autoencoder model class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the data from JSON file
file_path = '/Users/paullemaire/Documents/data2024.json'
df = pd.read_json(file_path, lines=True)

# Filter data based on timestamp
df = df[df['Downlink-GTW-Timestamp'] <= '2024-01-29 00:00:00']

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Adding the difference variable
df1 = df.sort_values(['Downlink-CRM-DeviceID', 'Downlink-GTW-Timestamp'])
df1['diff_seconds'] = df1.groupby('Downlink-CRM-DeviceID')['Downlink-GTW-Timestamp'].diff().dt.total_seconds()

# Define the function to pad or truncate sequences
def pad_or_truncate(seq, target_length):
    return seq[:target_length] + [0] * (target_length - len(seq))

# Initialize the autoencoder model
model = Autoencoder()

# Load the saved model state dictionary
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

# Define the loss criterion
criterion = nn.MSELoss()

# Compute the loss for each device and store the results
device_losses = {}
for device_id in df['Downlink-CRM-DeviceID'].unique():
    # Filter the data for the specified device
    df_device = df[df['Downlink-CRM-DeviceID'] == device_id]

    # Create the list of time differences
    time_diff = df_device['Downlink-GTW-Timestamp'].diff().dt.total_seconds().tolist()

    # Remove the initial NaN value
    time_diff = [x for x in time_diff if pd.notna(x)]

    if time_diff:
        # Normalize time_diff
        time_diff = [(i - min(time_diff)) / (max(time_diff) - min(time_diff)) for i in time_diff]

        # Pad or truncate to length 100
        time_diff_padded = pad_or_truncate(time_diff, 100)
        tensor = torch.tensor(time_diff_padded).float().unsqueeze(0)

        # Compute the reconstruction and loss
        with torch.no_grad():
            recon = model(tensor)
            loss = criterion(recon, tensor)
            device_losses[device_id] = loss.item()

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

# Sort the losses
sorted_device_losses = dict(sorted(device_losses.items(), key=lambda item: item[1]))

# Print the total number of devices evaluated
print(f'Total number of devices evaluated: {len(device_losses)}')

# Plot the sorted losses
plt.figure(figsize=(12, 6))
plt.bar(sorted_device_losses.keys(), sorted_device_losses.values())
plt.xticks(rotation=90)
plt.xlabel('Device ID')
plt.ylabel('Loss')
plt.title('Reconstruction Loss for Each Device (Sorted)')
plt.show()
