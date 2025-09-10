#%%
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os
#%%
# Step 1: Load the clean dataset
data_path = '../1.Preprocessing/clean_data.csv'
data = pd.read_csv(data_path)
#%%
# Separate features and target
X = data.drop('family', axis=1)  # Features
y = data['family']  # Target
#%%
# Binarize the output (for multi-class ROC curve)
classes = np.unique(y)
y_bin = label_binarize(y, classes=classes)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_test_bin = label_binarize(y_test, classes=classes)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Initialize classifiers
classifiers = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),  # For AUC calculation
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}

# Directory to store results
output_folder = '../7.Result/'  # Changed folder to '../7.Result/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Dictionary to store results
results = []

# Step 5: Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")
    
    # Record training time
    start_train = time.time()
    clf.fit(X_train_scaled, y_train)
    end_train = time.time()
    
    # Record testing time
    start_test = time.time()
    y_pred = clf.predict(X_test_scaled)
    end_test = time.time()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Compute AUC score for multi-class
    y_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None
    auc_score = roc_auc_score(y_test_bin, y_proba, multi_class='ovr') if y_proba is not None else None
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')  # Sensitivity
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Compute specificity for multi-class
    tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    specificity = np.mean(tn / (tn + cm.sum(axis=1) - np.diag(cm)))
    
    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc_score,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F1 Score": f1,
        "Training Time (s)": end_train - start_train,
        "Testing Time (s)": end_test - start_test
    })
    
    # Plot and save confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()  # Ensure everything fits well
    cm_image_path = os.path.join(output_folder, f'confusion_matrix_{name}.png')
    plt.savefig(cm_image_path)  # Save the confusion matrix image
    
    # Display the confusion matrix
    plt.show()
    plt.close()  # Close the plot to prevent overlapping figures
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {end_train - start_train:.4f} seconds")
    print(f"Testing Time: {end_test - start_test:.4f} seconds\n")

# Step 6: Plot ROC Curves for all classifiers with AUC scores
plt.figure(figsize=(10, 8))

for name, clf in classifiers.items():
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test_scaled)
        
        # Plot ROC curve for each class
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} Class {classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Classifiers (Multi-Class)')
plt.legend(loc="lower right")
plt.grid(True)

# Save the ROC curve plot
roc_image_path = os.path.join(output_folder, 'roc_curve_comparison.png')
plt.savefig(roc_image_path)
plt.show()

# Step 7: Create a DataFrame for results and display
results_df = pd.DataFrame(results)
print("\nSummary of Model Performance:")
print(results_df)

# Step 8: Save the results to an Excel file
results_excel_path = os.path.join(output_folder, 'model_performance_summary.xlsx')
results_df.to_excel(results_excel_path, index=False)

print(f"Results saved in: {output_folder}")
