import pandas as pd
import numpy as np

# Read the CSV file
file_path = 'output.csv'  # Replace 'your_file_path.csv' with the actual path to your CSV file
data = pd.read_csv(file_path,skiprows=[0])

# Shuffle the DataFrame
data_shuffled = data.sample(frac=1, random_state=42)  # Use a specific random_state for reproducibility

# Save the shuffled data as a new CSV file
output_file_path = 'output2.csv'  # Replace 'shuffled_data.csv' with your desired output file name
data_shuffled.to_csv(output_file_path, index=False)

print("Shuffled data saved to", output_file_path)
