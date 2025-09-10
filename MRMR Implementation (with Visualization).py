#%%
ستيراد المكتبات اللازمة
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import os

pandas: لتحليل البيانات ومعالجتها.
SelectKBest و mutual_info_classif: لاختيار أفضل الخصائص بناءً على درجات أهمية المعلومات.
matplotlib.pyplot: لإنشاء الرسومات البيانية.
os: لإدارة المسارات والملفات.




#%%
# Load the clean dataset
data_path = '../1.Preprocessing/clean_data.csv'
data = pd.read_csv(data_path)
يتم تحميل مجموعة البيانات من الملف الموجود في المسار ../1.Preprocessing/clean_data.csv باستخدام pandas.
#%%

# Separate features and target
X = data.drop('family', axis=1)
y = data['family']
X: يحتوي على الخصائص (features) عن طريق حذف العمود family.
y: يحتوي على العمود الهدف (target) الذي يمثل فئة البرمجية الخبيثة أو الفدية.
#%%


# Apply MRMR (using SelectKBest with mutual information as a proxy for MRMR)
selector = SelectKBest(score_func=mutual_info_classif, k='all')  # Compute scores for all features
selector.fit(X, y)
MRMR (Minimum Redundancy Maximum Relevance): تُستخدم لتحديد أفضل الخصائص من حيث صلتها بالهدف مع تقليل التكرار بينها.
SelectKBest:
تُطبق دالة mutual_info_classif التي تعتمد على حساب المعلومات المتبادلة بين الخصائص والهدف لتحديد الأهمية.
يتم حساب درجات الأهمية لجميع الخصائص.


# Get the scores for all features
scores = selector.scores_
features = X.columns

# Create a DataFrame to display features and their scores
features_scores = pd.DataFrame({
    'Feature': features,
    'Score': scores
})

# Sort features by score in descending order and select the top 20 features
top_20_features = features_scores.sort_values(by='Score', ascending=False).head(40)
scores: تخزن درجات الأهمية لكل خاصية.
features: أسماء الأعمدة (الخصائص) في مجموعة البيانات.
features_scores: DataFrame يحتوي على الخصائص ودرجاتها.
top_20_features: اختيار أعلى 20 خاصية من حيث الدرجات.



# Display the scores for the top 20 features
print("\nTop 20 Features by MRMR Score:")
print(top_20_features)

# Visualize the top 20 features
plt.figure(figsize=(10, 8))
plt.barh(top_20_features['Feature'], top_20_features['Score'], color='skyblue')
plt.xlabel('Mutual Information Score')
plt.ylabel('Top 20 Features')
plt.title('Top 20 Features Selected by MRMR')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.grid(True)
يتم رسم رسم بياني شريطي أفقي يعرض أعلى 20 خاصية مع درجاتها.
invert_yaxis: لعكس ترتيب الخصائص بحيث تظهر الأهم أولاً.







# Save the MRMR plot for the top 20 features
output_folder = '../4.FS_MRMR/'
plt.savefig(os.path.join(output_folder, 'mrmr_top_20_features.png'))
plt.show()

# Save top 20 features in a CSV file
top_20_features.to_csv(os.path.join(output_folder, 'top_20_features.csv'), index=False)
يتم حفظ الرسم البياني كصورة في المجلد ../4.FS_MRMR/.
يتم حفظ بيانات الخصائص العشرين الأولى كملف CSV ليسهل الرجوع إليها لاحقاً.
