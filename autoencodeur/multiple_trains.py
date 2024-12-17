import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random

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

# Process data to get final_data and aux_data
final_data = {}
aux_data = []
for i in df1['Downlink-CRM-DeviceID'].unique():
    device_data = df1[df1['Downlink-CRM-DeviceID'] == i]['diff_seconds']
    value = round(device_data.std() / device_data.mean(), 2)
    if not value in final_data.keys():
        final_data[value] = 0
    final_data[value] += 1
    aux_data.append((i, value))

final_data = dict(sorted(final_data.items(), key=lambda x: x[0]))
aux_data = list(sorted(aux_data, key=lambda x: x[1]))[::-1]
List_of_devices = [aux_data[i][0] for i in range(len(aux_data)) if aux_data[i][1] <= 0.35]

# Function to pad or truncate sequences
def pad_or_truncate(seq, target_length):
    return seq[:target_length] + [0] * (target_length - len(seq))

# Create list of tensors
List_of_tensors = []
for device in List_of_devices:
    # Filter the data for the specified device
    df_device = df[df['Downlink-CRM-DeviceID'] == device]

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
        List_of_tensors.append(tensor)

# Define the autoencoder model
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

# Initialize model, criterion and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 300000
outputs = []
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Shuffle the list of tensors at the start of each epoch
    random.shuffle(List_of_tensors)
    
    tensor = List_of_tensors[epoch % len(List_of_tensors)]
    recon = model(tensor)
    loss = criterion(recon, tensor)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % (num_epochs // 10) == 0:
        print(f'Epoch:{epoch}, Loss:{loss.item()}')
    outputs.append((epoch, tensor, recon))

# Plot the results in 4 subplots
def plot_reconstruction_subplots(outputs, losses, num_epochs):
    fig, axs = plt.subplots(2, 2, figsize=(15, 7))
    indices = [0, num_epochs // 3, 2 * num_epochs // 3, num_epochs - 1]
    for ax, k in zip(axs.flatten(), indices):
        original = outputs[k][1][0].detach().numpy()
        reconstructed = outputs[k][2][0].detach().numpy()
        ax.plot(original, label='Original')
        ax.plot(reconstructed, label='Reconstructed')
        ax.set_title(f'Epoch {outputs[k][0]}, Loss={losses[k]:.4f}')
        ax.legend()
    plt.tight_layout()
    plt.show()

plot_reconstruction_subplots(outputs, losses, num_epochs)

# save the model
torch.save(model.state_dict(), 'autoencoder2.pth')