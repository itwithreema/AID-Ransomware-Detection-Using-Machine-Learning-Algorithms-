#%%
import pandas as pd
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
pandas: لإدارة وتحليل البيانات.
os: لإدارة الملفات والمجلدات.
SelectKBest و mutual_info_classif: لاختيار أفضل الخصائص باستخدام خوارزمية MRMR (Minimum Redundancy Maximum Relevance).



# Load the cleaned dataset from the previous step
data_path = '../1.Preprocessing/clean_data.csv'
data = pd.read_csv(data_path)
يتم تحميل مجموعة البيانات المُنظَّفة التي تم تجهيزها في مرحلة سابقة من المسار ../1.Preprocessing/clean_data.csv.



# Separate features and target
X = data.drop('family', axis=1)
y = data['family']
: يحتوي على الخصائص (features) عن طريق حذف العمود family.
y: يحتوي على العمود الهدف (target) وهو فئة البرمجيات الخبيثة أو الفدية




# Step 1: Apply MRMR (using SelectKBest with mutual information as a proxy for MRMR)
print("Applying MRMR to select the top 20 features...")
selector = SelectKBest(score_func=mutual_info_classif, k=20)  # Select top 20 features
X_new = selector.fit_transform(X, y)
 الدالتين نفسهم و الجديدة النيو x

# Get the selected feature names
selected_features = X.columns[selector.get_support()]
get_support(): تُعيد مصفوفة تحتوي على القيم True للخصائص التي تم اختيارها.
selected_features: قائمة بأسماء الخصائص العشرين المختارة.


# Step 2: Create a new DataFrame with the selected features
data_mrmr = pd.DataFrame(X_new, columns=selected_features)
data_mrmr['family'] = y  # Add the target column back
يتم إنشاء DataFrame جديد يحتوي فقط على أفضل 20 خاصية مع إضافة العمود الهدف family.




# Step 3: Save the new dataset with the top 20 MRMR features
output_folder = '../1.Preprocessing/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

clean_data_mrmr_path = os.path.join(output_folder, 'clean_data_mrmr.csv')
data_mrmr.to_csv(clean_data_mrmr_path, index=False)
os.path.exists: يتحقق مما إذا كان المجلد موجودًا.
os.makedirs: إذا لم يكن المجلد موجودًا، يتم إنشاؤه.
to_csv: يتم حفظ مجموعة البيانات الجديدة التي تحتوي على أفضل 20 خاصية في ملف clean_data_mrmr.csv.






print(f"Dataset with the top 20 MRMR features has been saved as '{clean_data_mrmr_path}'")
رسالة تُخبرك بأنه تم حفظ مجموعة البيانات بنجاح.
