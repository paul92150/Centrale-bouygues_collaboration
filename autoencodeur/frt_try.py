import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Load the data from JSON file
file_path = '/Users/paullemaire/Documents/data2024.json'
df = pd.read_json(file_path, lines=True)

device = 'be6fcf9f-060a-41d1-ac34-be97b9d72f10'

df = df[df['Downlink-GTW-Timestamp'] <= '2024-01-29 00:00:00']

# Filter the data for the specified device
df_device = df[df['Downlink-CRM-DeviceID'] == device]

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df_device['Downlink-GTW-Timestamp'] = pd.to_datetime(df_device['Downlink-GTW-Timestamp'])

# Create the list of time differences
time_diff = df_device['Downlink-GTW-Timestamp'].diff().dt.total_seconds().tolist()

# Remove the initial NaN value
time_diff = [x for x in time_diff if pd.notna(x)]

# Normalize time_diff
time_diff = [(i - min(time_diff)) / (max(time_diff) - min(time_diff)) for i in time_diff]

# Pad or truncate to length 100
def pad_or_truncate(seq, target_length):
    return seq[:target_length] + [0]*(target_length - len(seq))

# Prepare the data
time_diff_padded = pad_or_truncate(time_diff, 100)
tensor = torch.tensor(time_diff_padded).float().unsqueeze(0)

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

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
outputs = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    recon = model(tensor)
    loss = criterion(recon, tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % (num_epochs//10) == 0:
        print(f'Epoch:{epoch}, Loss:{loss.item()}')
    outputs.append((epoch, tensor, recon))

# Plot the results
def plot_reconstruction(original, reconstructed):
    plt.figure(figsize=(12, 6))
    plt.plot(original[0].detach().numpy(), label='Original')
    plt.plot(reconstructed[0].detach().numpy(), label='Reconstructed')
    plt.legend()
    plt.show()

for k in range(0, num_epochs,num_epochs//4):
    plot_reconstruction(outputs[k][1], outputs[k][2])

# Save the model
#torch.save(model.state_dict(), 'autoencoder.pth')
