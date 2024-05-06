import pandas as pd
import numpy as np

# Sample data
data = pd.read_csv('./data/rf_training/simulation_episodes.csv')

# Create DataFrame
df = pd.DataFrame(data)

# Filter out extreme samples
df_filtered = df[(df['MappedRCP'] > df['MappedRCP'].quantile(0.05)) & 
                 (df['MappedRCP'] < df['MappedRCP'].quantile(0.95))]

# 2. Shuffle MappedRCP values and randomly assign them to a new truck with a new truck_id
# Shuffle MappedRCP values
shuffled_values = np.random.permutation(df_filtered['MappedRCP'])

# Assign shuffled values to a new column
df_filtered.loc[:, 'MappedRCP_Shuffled'] = shuffled_values

# Generate sequential truck ids
df_filtered['Truckids'] = range(1, len(df_filtered) + 1)

# 3. Create a DataFrame with the required structure
# Define tire mappings
tire_mappings = { 
    'Truckids': ['Truckids'],
    'steertireleft': ['MappedRCP_Shuffled'],
    'steertireright': ['MappedRCP_Shuffled'],
    'drive1tireleft': ['MappedRCP_Shuffled'],
    'drive1tirelright': ['MappedRCP_Shuffled'],
    'drive2tireleft': ['MappedRCP_Shuffled'],
    'drive2tirelright': ['MappedRCP_Shuffled'],
    'rear1tireleft': ['MappedRCP_Shuffled'],
    'rear1tirelright': ['MappedRCP_Shuffled'],
    'rear2tireleft': ['MappedRCP_Shuffled'],
    'rear2tirelright': ['MappedRCP_Shuffled']
}

# Create the final DataFrame
final_df = pd.DataFrame(columns=tire_mappings.keys())

# Fill the final DataFrame with shuffled values
for column, values in tire_mappings.items():
    final_df[column] = df_filtered.sample(frac=1)[values[0]].values
    
#sort the dataframe by truckid
final_df = final_df.sort_values(by=['Truckids'])

# Save the final DataFrame to a CSV file
final_df.to_csv('./data/rf_training/final_data.csv', index=False)
