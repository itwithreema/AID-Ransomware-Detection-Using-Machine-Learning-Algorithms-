#%%
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
#%%
# Read the dataset from CSV file
data_path = '../0.dataset/data.csv'
data = pd.read_csv(data_path)
#%%
# Step 1: Check for missing values
print("Checking for missing values...")
missing_values = data.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nThere are missing values. You may need to handle them before proceeding.")
#%%
# Step 2: Check for duplicate rows
print("\nChecking for duplicate rows...")
duplicate_rows = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

if duplicate_rows > 0:
    # Drop duplicate rows
    data = data.drop_duplicates()
    print(f"Duplicate rows have been removed. New shape: {data.shape}")
#%%
# Step 3: Label encode the 'family' column
print("\nLabel encoding 'family' column...")
label_encoder = LabelEncoder()
data['family'] = label_encoder.fit_transform(data['family'])
#%%
# Step 4: Check class distribution (before resampling)
print("\nClass distribution:")
class_distribution = data['family'].value_counts()
print(class_distribution)
#%%
# Step 6: Check for missing values in cleaned data
print("\nChecking for missing values in the cleaned data...")
cleaned_missing_values = data.isnull().sum()
print(cleaned_missing_values)

# If there are missing values, drop rows with missing data
if cleaned_missing_values.sum() > 0:
    print("\nThere are missing values in the cleaned data. Dropping rows with missing values...")
    data_scaled = data.dropna()
    print(f"New shape after dropping missing values: {data_scaled.shape}")
else:
    print("\nNo missing values found in the cleaned data.")
#%%
# Step 5: Save the cleaned dataset
output_folder = '../1.Preprocessing/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

clean_data_path = os.path.join(output_folder, 'clean_data.csv')
data.to_csv(clean_data_path, index=False)

print(f"Cleaned dataset has been saved as '{clean_data_path}'")
#%%