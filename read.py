
#import necessary library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
#%%
# Read the dataset from CSV file
data_path = '../0.dataset/data.csv'
data = pd.read_csv(data_path)
#%%
# Label encoding the 'family' column
label_encoder = LabelEncoder()
data['family'] = label_encoder.fit_transform(data['family'])
#%%
# Basic data analysis
print("Dataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

# Statistical description of the dataset
description = data.describe()
print("\nStatistical Description:")
print(description)

# Save statistical description to a file
output_folder = '../1.Preprocessing/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

description.to_csv(os.path.join(output_folder, 'data_description.csv'))
#%%
# Data visualization and plotting
sns.set(style="whitegrid")

# 1. Class distribution in 'family' column (after encoding)
plt.figure(figsize=(10, 6))
sns.countplot(x='family', data=data)
plt.title('Class Distribution in the Dataset')

# Save the class distribution plot
plt.savefig(os.path.join(output_folder, 'family_distribution.png'))
plt.show()

# 2. Correlation Heatmap (after encoding 'family')
plt.figure(figsize=(30, 30))
correlation_matrix = data.corr()  # Calculate the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')

# Save the heatmap
plt.savefig(os.path.join(output_folder, 'correlation_heatmap.png'))
plt.show()
#%%
# 3. Distribution of numeric features
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of values in {column}')
    
    # Save the distribution plots
    plt.savefig(os.path.join(output_folder, f'distribution_{column}.png'))
    plt.show()
#%%
# 4. Pairplot analysis of numeric features
plt.figure(figsize=(15, 15))
sns.pairplot(data[numeric_columns], diag_kind='kde', markers='+')
plt.title('Pairplot Analysis of Numeric Features')

# Save the pairplot analysis
plt.savefig(os.path.join(output_folder, 'pairplot_analysis.png'))
plt.show()
#%%