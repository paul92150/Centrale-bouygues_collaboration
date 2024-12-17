import matplotlib.pyplot as plt
import pandas as pd

List_of_high_CV = [('4e1ad3a9-c947-4a99-9bbe-6a01c669bc53', 1.4), ('f1aa0c14-7014-4491-acff-82953c519abc', 1.38), ('5e3b87d4-640f-4234-9b20-60620429cbf3', 1.08), ('8ebb2dbe-f83d-42d1-86e2-073392f7a343', 1.06), ('455875d0-b14e-47ed-a8f2-f3bdd9ae23ff', 1.04), ('e66f2425-2499-493f-a5a4-ff3b2d3eff51', 1.02), ('8e18b765-f76a-44ff-956e-5af377d94621', 1.02), ('49594d0d-8d1f-4add-a559-f9b63f740037', 0.95), ('0510b4df-b0be-4dc4-b81e-2b288516fd6b', 0.93), 
                   ('67ddd3a1-d630-49b1-83aa-2f87053bd513', 0.89), ('9e3fd9ab-dbe6-4b83-b367-11df51d6512d', 0.85), ('61fbaddf-5461-4f2e-bb49-1abe9c77aba3', 0.85), ('3f8809cd-3de4-4a98-bc4f-752213b8af39', 0.85), ('7f48cf4d-daa3-499b-9866-9caa128805bf', 0.82), ('61d1dfad-5c94-45c5-9618-4f92f69ccc4e', 0.8), ('ab39b18a-9ad9-413b-b436-9678d4be4bc0', 0.79), ('6dbc170f-b1a5-4668-8004-cc7aa9fbdc2a', 0.79), ('5839e957-47b5-4628-8430-03a1ef611a2c', 0.78), 
                   ('5416bd19-2830-4c81-bca9-63c3ab7fefce', 0.77), ('d63f2c8b-8be7-4809-b496-b0fa463549bb', 0.75), ('c3f68b78-e1b5-475e-a606-dd00fcb95fa7', 0.74), ('5cfd2ab4-074c-4cf6-91e2-0beb5b5cfcbe', 0.71), ('2ad4c9f5-34b7-4ab4-8e13-6ef548cad5e5', 0.71), ('70c38b57-79fb-4323-b92d-fdeccbdf05cc', 0.65), ('aef7a909-6f1f-4ae5-9d03-f3e17b28344f', 0.64), ('654a3bb7-5669-4d42-b37d-50e153a9d1e4', 0.64), ('166ceebe-f42f-49a1-bf42-73697942238e', 0.6), 
                   ('50de6e02-c915-4fc6-aa26-fe2ba6598c57', 0.58), ('de7e208c-cd6c-4946-b384-74064878123c', 0.56), ('69f225de-5d9b-4bff-9214-dbb0881a31ce', 0.56), ('32be0c37-a9b1-4de2-a17a-bba8a87b61d0', 0.54), ('6304483c-ddd0-428f-b43b-20c1a0cef904', 0.53), ('1200c9e3-091c-4634-af71-0503567b2271', 0.51), ('4ee5f1a2-042f-43c0-851d-f45f5fb05dac', 0.49), ('09c7950a-0b23-4d1c-91f3-502c8a14444a', 0.47), ('aaf98747-fa11-4aa0-a4f0-40b23c6b09c3', 0.4)]

List_of_low_CV = [('e5165d7f-36ce-44b5-b3b3-dc4045739c31', 0.01), ('e26ceb58-134a-4b8b-a190-5ffac855521d', 0.01), ('98a3f34a-8309-4464-982b-69d85ad56e04', 0.14), ('8c50f67e-11fd-46dd-a4f0-90c88d08d0c4', 0.15),('db6edc21-a42e-4049-a4eb-b8b2fb54984e', 0.01), ('d1e479c0-e9ff-496e-ad78-70a94a9a125b', 0.01), ('d0a2ce2b-4eee-461e-a4b3-8f8e7ce86ed1', 0.01), ('cc25d9ec-68d5-4a95-9c32-880888853a58', 0.01), ('cadf5d4a-7f4b-467e-a425-c2f8f8c53ea9', 0.01), ('c7923ce8-9efd-4ba4-a621-7a37f25c3e6d', 0.01), 
                  ('b92ded1b-bbd9-4aa4-bf26-2c66612e0357', 0.01), ('a00e8c56-1b5b-4369-a323-e0cedfeaa1a8', 0.01), ('8b7d5cc3-d056-4b5c-addf-3ad3fe8dc2b4', 0.01), ('850831b9-5093-4256-965d-fae98867334a', 0.01), ('84d6d5e7-1478-4827-8a70-fe3ba4a5a347', 0.01), ('7f32cbf8-cd29-4ee6-84d6-be4edc51f30f', 0.01), ('790d22b5-d1ed-4bc9-8205-22759480b7a3', 0.01), ('73b94708-0fb0-4e7f-bdc8-f53dcb2417ce', 0.01), ('68c29f16-ef9f-4119-870e-da6d99e4004f', 0.01), 
                  ('5d73867e-d491-41e8-9561-57c3b7916bc5', 0.01), ('5718c30c-04fb-4c75-a1e0-d3f9a58239c4', 0.01), ('4c26981a-d33c-4978-808d-054269eeb3a9', 0.01), ('43b97bd3-c7cb-4a8f-b91e-6f93d7eb6792', 0.01), ('42efe95e-8d1a-47ad-b414-3b076ba0b211', 0.01), ('3b1ff25f-06ce-437f-ba25-420c539d15c9', 0.01), ('3876520b-a583-48b1-b833-aa66d2c5770c', 0.01), ('376b0ae5-28f1-4ca3-b950-3018243785f1', 0.01), ('311d4367-4acc-491b-98ee-08b94b66560c', 0.01), 
                  ('2a3d16d1-c00c-4008-b772-99aba00473c5', 0.01), ('17c6b041-5962-4a5f-9b4d-437cff076264', 0.01), ('154b05cc-9b8d-48e6-afe2-8336c042a135', 0.01), ('1182386f-d653-49f8-946f-b5d0400893f2', 0.01)]

# List_of_high_CV sorted by CV
List_of_high_CV = sorted(List_of_high_CV, key=lambda x: x[1], reverse=True)

# Load the data from JSON file with error handling
file_path = '/Users/paullemaire/Documents/data2024.json'
try:
    df = pd.read_json(file_path, lines=True)
except ValueError as e:
    print(f"Error reading JSON file: {e}")
    exit(1)

# Convert the 'Downlink-GTW-Timestamp' column to datetime
df['Downlink-GTW-Timestamp'] = pd.to_datetime(df['Downlink-GTW-Timestamp'])

# Extract date for filtering
df['date'] = df['Downlink-GTW-Timestamp'].dt.date

# Define the dates of interest and device ID
dates_of_interest = [pd.to_datetime('2024-01-27').date(),
                     pd.to_datetime('2024-01-28').date()]

def afficher_graphique(device_id, device_CV, ax):

    # Filter rows for the specified device ID
    df_device = df[df['Downlink-CRM-DeviceID'] == device_id]

    # Check if the DataFrame is empty before proceeding
    if df_device.empty:
        print("No data available for the specified device ID.")
    else:
        # Combine data for consecutive days
        for i in range(len(dates_of_interest)-1):
            start_date = dates_of_interest[i]
            end_date = dates_of_interest[i+1]
            df_period = df_device[(df_device['date'] == start_date) | (df_device['date'] == end_date)]
            
            if df_period.empty:
                print(f"No data available for the period {start_date} to {end_date}.")
                continue
            
            # Sort by timestamp
            df_period = df_period.sort_values('Downlink-GTW-Timestamp')
            
            # Calculate time difference between consecutive responses
            df_period['time_diff'] = df_period['Downlink-GTW-Timestamp'].diff().dt.total_seconds()
            
            # Drop the first row with NaN time difference
            df_period = df_period.dropna(subset=['time_diff'])

            # Calculate average time difference
            avg_time_diff = df_period['time_diff'].mean()
            std_time_diff = df_period['time_diff'].std()

            # Coefficient of variation
            cv = std_time_diff / avg_time_diff
            print(f"CV: {cv}")
            
            # Plot the time differences
            for index, row in df_period.iterrows():
                color = 'red' if row['time_diff'] > 2 * avg_time_diff else 'green'
                ax.plot(row['Downlink-GTW-Timestamp'], row['time_diff'], marker='o', color=color)

            ax.set_title(f'Device ID: {device_id} with CV: {device_CV}')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Time Difference (s)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        ax.legend()
n = 10

List_of_high_CV
fig, axs = plt.subplots(n // 2 + n % 2, 2, figsize=(14, 80/n), sharey=True)
axs = axs.flatten()

for i in range(6):
    afficher_graphique(List_of_high_CV[i][0], List_of_high_CV[i][1], axs[i])
for i in range(4):
    afficher_graphique(List_of_low_CV[i][0], List_of_low_CV[i][1], axs[i+6])

# Hide any unused subplots
for j in range(n, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()
