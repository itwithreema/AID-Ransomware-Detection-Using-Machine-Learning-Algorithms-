import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Step 1: Load the cleaned dataset
data_path = '../1.Preprocessing/clean_data.csv'
data = pd.read_csv(data_path)

# Separate features and target
X = data.drop('family', axis=1)  # Features
y = data['family']  # Target

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
# Fit PCA on the scaled data without specifying the number of components (retain all components)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 4: Visualize explained variance to decide how many components to choose
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)  # Cumulative explained variance

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--', color='b')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='-')  # Line indicating 95% of variance explained
plt.text(0.5, 0.9, '95% Explained Variance Threshold', color='red', fontsize=12)
plt.show()

# Step 5: Select the number of components that explain 95% of the variance
n_components = np.argmax(explained_variance_ratio >= 0.95) + 1
print(f"\nNumber of components that explain 95% of the variance: {n_components}")

# Step 6: Refit PCA with the chosen number of components
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

# Step 7: Create a new DataFrame with the reduced data
data_reduced = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(n_components)])
data_reduced['family'] = y  # Add the target column back

# Step 8: Save the new reduced dataset
output_folder = '../1.Preprocessing/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

reduced_data_path = os.path.join(output_folder, 'clean_data_pca.csv')
data_reduced.to_csv(reduced_data_path, index=False)

print(f"Reduced dataset has been saved as '{reduced_data_path}'")
