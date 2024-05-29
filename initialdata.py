import pandas as pd

# Example data for two trucks with ten tires each
data = {
    'Tire 1': [0.8, 0.9],
    'Tire 2': [0.7, 0.8],
    'Tire 3': [0.9, 0.7],
    'Tire 4': [0.6, 0.9],
    'Tire 5': [0.5, 0.6],
    'Tire 6': [0.4, 0.5],
    'Tire 7': [0.7, 0.7],
    'Tire 8': [0.8, 0.8],
    'Tire 9': [0.9, 0.9],
    'Tire 10': [1.0, 0.9]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
file_path = 'initial_state.xlsx'
df.to_excel(file_path, index=False)

print(f"Excel file created: {file_path}")