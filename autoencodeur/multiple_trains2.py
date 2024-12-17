import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Load the data from JSON file
file_path = '/Users/paullemaire/Documents/data2024.json'
df = pd.read_json(file_path, lines=True)

# Filter data based on timestamp
df = df[df['Downlink-GTW-Timestamp'] <= '2024-01-29 00:00:00']

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Adding the difference variable
df = df.sort_values(['Downlink-CRM-DeviceID', 'Downlink-GTW-Timestamp'])
df['diff_seconds'] = df.groupby('Downlink-CRM-DeviceID')['Downlink-GTW-Timestamp'].diff().dt.total_seconds()

# Process data to get final_data and aux_data
final_data = {}
aux_data = []
for i in df['Downlink-CRM-DeviceID'].unique():
    device_data = df[df['Downlink-CRM-DeviceID'] == i]['diff_seconds']
    value = round(device_data.std() / device_data.mean(), 2)
    if value not in final_data:
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

# Split data into training and validation sets
train_tensors, val_tensors = train_test_split(List_of_tensors, test_size=0.2, random_state=42)

# Create data loaders
batch_size = 32
train_loader = DataLoader(TensorDataset(torch.cat(train_tensors)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.cat(val_tensors)), batch_size=batch_size, shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 100),
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

# Early stopping and learning rate scheduler
early_stopping_patience = 100
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


# Training loop with early stopping
num_epochs = 900
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

def calculate_average_loss_on_devices(model, List_of_tensors):
    model.eval()
    total_loss = 0.0
    num_devices = 0
    with torch.no_grad():
        for tensor in List_of_tensors:
            recon = model(tensor)
            loss = criterion(recon, tensor)
            total_loss += loss.item()
            num_devices += 1

    if num_devices == 0:
        return float('inf')
    return total_loss / num_devices

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        tensor = batch[0]
        recon = model(tensor)
        loss = criterion(recon, tensor)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            tensor = batch[0]
            recon = model(tensor)
            loss = criterion(recon, tensor)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    average_loss_on_devices = calculate_average_loss_on_devices(model, List_of_tensors)

    if average_loss_on_devices < best_val_loss:
        best_val_loss = average_loss_on_devices
        patience_counter = 0
        torch.save(model.state_dict(), 'best_autoencoder2.pth')
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch}')
        break

    if epoch % (num_epochs // 10) == 0:
        print(f'Epoch:{epoch}, Train Loss:{train_loss}, Val Loss:{val_loss}, Avg Loss on Devices:{average_loss_on_devices}')

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
