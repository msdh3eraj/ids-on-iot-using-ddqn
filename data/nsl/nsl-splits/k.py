import pandas as pd

# Define the input and output file paths
input_file = 'set-1.csv'  # Replace with your input file path
output_file = 'output.csv'  # Replace with your desired output file path

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(input_file)

# Map 'dom' to 1 and 'normal' to 0 in the last column
data.iloc[:, -1] = data.iloc[:, -1].map({'dos': 1, 'normal': 0})

# Save the modified data to an output CSV file
data.to_csv(output_file)
