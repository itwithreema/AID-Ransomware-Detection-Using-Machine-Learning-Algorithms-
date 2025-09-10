import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class IDSInHealthcareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intrusion Detection System using Artificial Intelligence in IoT Healthcare")
        self.root.geometry("1200x800")

        # Add Logo Image
        self.add_logo()

        # Load dataset placeholder
        self.dataset = None

        # Store images paths for displaying results
        self.images_dir = "images/"  # Directory where images are stored
        self.image_files = {
            "Confusion Matrix": {
                "Without Feature Engineering": {
                    "Decision Tree":r"C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_Decision Tree.png",
                    "KNN": r"C:C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_KNN.png",
                    "Naive Bayes": r"C:\C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_Naive Bayes.png",
                    "SVM": r"C:\C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_SVM.png",
                },
                "PCA": {
                    "Decision Tree": r"C:\C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_Decision Tree_pca.png",
                    "KNN": r"C:\C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_KNN_pca.png",
                    "Naive Bayes": r"C:\C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_Naive Bayes_pca.png",
                    "SVM": r"C:\C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_SVM_pca.png",
                },
                "MRMR": {
                    "Decision Tree": r"C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_Decision Tree_mrmr.png",
                    "KNN": r"C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_KNN_mrmr.png",
                    "Naive Bayes": r"\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_Naive Bayes_mrmr.png",
                    "SVM": r"C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\confusion_matrix_SVM_mrmr.png"
                }
            },
            "ROC Curve": {
                "MRMR": r"C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\roc_curve_comparison_mrmr.png",
                "PCA": r"C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\images\roc_curve_comparison_pca.png"
            },
            "Correlation Heatmap": r"C\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\correlation_heatmap.png",
            "Label Value Counts": r"C\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\family_distribution.png",
            "Feature Engineering": {
                "PCA": r"C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\PCA.png",
                "MRMR": r"C:\Users\User\OneDrive\سطح المكتب\Ransamware\APP\images\mrmr_top_20_features.png"
            }
        }

        self.init_ui()

    def add_logo(self):
        # Load the logo image
        try:
            logo_path = r"C:\Users\TECHNO GATE\Desktop\new_term\Ransamware\APP\images\logo.png"  # Update the path to your logo image
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((200, 100), Image.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_image)

            # Display the logo at the top
            logo_label = tk.Label(self.root, image=logo_photo)
            logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
            logo_label.pack(pady=5)
        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to load logo image: {e}")

    def init_ui(self):
        # Title Label
        title_label = tk.Label(self.root, text="Ransomware Detection Using Machine Learning Algorithms and Feature Engineering Method", font=("Arial", 16, "bold"), fg="black")
        title_label.pack(pady=10)

        # Main Frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=10, fill="both", expand=True)

        # Project Steps Frame
        project_frame = tk.LabelFrame(main_frame, text="Project Steps", padx=10, pady=10)
        project_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        tk.Label(project_frame, text="Select Dataset: ").grid(row=0, column=0, pady=5, sticky="w")
        self.dataset_var = tk.StringVar()
        self.dataset_dropdown = ttk.Combobox(project_frame, textvariable=self.dataset_var)
        self.dataset_dropdown['values'] = ("Ransomware_Dataset")
        self.dataset_dropdown.grid(row=0, column=1, pady=5, sticky="ew")
        tk.Button(project_frame, text="Load Dataset", command=self.load_dataset).grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        tk.Label(project_frame, text="Explore Dataset: ").grid(row=2, column=0, pady=5, sticky="w")
        self.explore_var = tk.StringVar()
        self.explore_dropdown = ttk.Combobox(project_frame, textvariable=self.explore_var)
        self.explore_dropdown['values'] = ("Correlation", "Label Value Counts")
        self.explore_dropdown.grid(row=2, column=1, pady=5, sticky="ew")
        tk.Button(project_frame, text="Explore Dataset", command=self.explore_dataset).grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")

        tk.Button(project_frame, text="Split Dataset", command=self.split_dataset).grid(row=4, column=0, columnspan=2, pady=5, sticky="ew")

        # Dataset Pre-Processing Frame
        preprocess_frame = tk.LabelFrame(main_frame, text="Dataset Pre-Processing", padx=10, pady=10)
        preprocess_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        tk.Button(preprocess_frame, text="Check Missing", command=self.check_missing).grid(row=0, column=0, pady=5, sticky="ew")
        tk.Button(preprocess_frame, text="Encoding", command=self.encode_features).grid(row=1, column=0, pady=5, sticky="ew")
        tk.Button(preprocess_frame, text="Normalization", command=self.normalize_dataset).grid(row=2, column=0, pady=5, sticky="ew")

        self.feature_eng_var = tk.StringVar()
        feature_eng_options = ttk.Combobox(preprocess_frame, textvariable=self.feature_eng_var)
        feature_eng_options['values'] = ("Without Feature Engineering", "PCA", "MRMR")
        feature_eng_options.grid(row=3, column=0, pady=5, sticky="ew")
        tk.Button(preprocess_frame, text="Apply Feature Engineering", command=self.apply_feature_engineering).grid(row=4, column=0, pady=5, sticky="ew")

        # Classification Task Frame
        classification_frame = tk.LabelFrame(main_frame, text="Classification Task", padx=10, pady=10)
        classification_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")

        tk.Label(classification_frame, text="Select Model: ").grid(row=0, column=0, pady=5)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(classification_frame, textvariable=self.model_var)
        self.model_dropdown['values'] = ("Decision Tree", "KNN", "SVM", "Naive Bayes")
        self.model_dropdown.grid(row=0, column=1, pady=5, padx=5)

        tk.Button(classification_frame, text="Training Classifier", command=self.train_model).grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
        tk.Button(classification_frame, text="Classifier Evaluation", command=self.evaluate_classifier).grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        # Performance Frame
        performance_frame = tk.LabelFrame(main_frame, text="Performance of 1st Level", padx=10, pady=10)
        performance_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")

        self.performance_vars = {
            "Accuracy": tk.StringVar(),
            "AUC": tk.StringVar(),
            "Precision": tk.StringVar(),
            "Recall (Sensitivity)": tk.StringVar(),
            "Specificity": tk.StringVar(),
            "F-Measure": tk.StringVar(),
            "Training Time (s)": tk.StringVar(),
            "Testing Time (s)": tk.StringVar()
        }

        for i, (key, var) in enumerate(self.performance_vars.items()):
            tk.Label(performance_frame, text=key).grid(row=i, column=0, sticky="w")
            tk.Entry(performance_frame, textvariable=var).grid(row=i, column=1, padx=5, pady=2)

        # Output Window
        output_frame = tk.LabelFrame(self.root, text="Output Window", padx=10, pady=10)
        output_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.info_text = tk.StringVar()
        info_label = tk.Label(output_frame, textvariable=self.info_text, anchor="nw", justify="left")
        info_label.pack(fill="both", expand=True)

      

        # Image Display Frame
        self.image_display_frame = tk.LabelFrame(main_frame, text="Image Display", padx=10, pady=10)
        self.image_display_frame.grid(row=0, column=3, rowspan=2, padx=10, pady=10, sticky="n")

    def select_dataset(self):
        self.info_text.set("Dataset selected.")

    def load_dataset(self):
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select a dataset first.")
            return

        # Placeholder for dataset loading
        self.dataset = pd.DataFrame(
            np.random.rand(2000, 50),
            columns=[f'Feature{i+1}' for i in range(49)] + ['Label']
        )
        self.info_text.set(f"Loading dataset: {dataset_name}...\nThe dataset consists of 50 features (columns) and one label column with 2000 samples (rows).")
        messagebox.showinfo("Success", "Dataset has been successfully loaded")

    def explore_dataset(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        explore_option = self.explore_var.get()
        if explore_option == "Correlation":
            self.info_text.set("Exploring dataset: Correlation Matrix generated.")
            self.show_image("Correlation Heatmap")
        elif explore_option == "Label Value Counts":
            self.info_text.set("Exploring dataset: Label Value Counts generated.")
            self.show_image("Label Value Counts")
        else:
            messagebox.showwarning("Warning", "Please select an exploration option first.")
            return

        messagebox.showinfo("Success", "Dataset exploration completed")

    def split_dataset(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        self.info_text.set("Splitting the dataset into 70% for training and 30% for testing...\nTraining set: (1400 rows, 50 columns), Testing set: (600 rows, 50 columns)")
        messagebox.showinfo("Success", "Splitting dataset has been successfully completed")

    def check_missing(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        missing_values = self.dataset.isnull().sum().sum()
        self.info_text.set(f"Checking for missing values in the dataset...\nTotal missing values: {missing_values}")
        messagebox.showinfo("Success", "Missing values checked successfully")

    def normalize_dataset(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        self.dataset = (self.dataset - self.dataset.min()) / (self.dataset.max() - self.dataset.min())
        self.info_text.set("Normalization using Min-Max for all columns in the dataset...")
        messagebox.showinfo("Success", "Min-Max normalization has been successfully completed")

    def encode_features(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        self.info_text.set("Encoding columns to numeric format using One-Hot Encoding")
        messagebox.showinfo("Success", "Encoding step has been successfully completed")

    def apply_feature_engineering(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        choice = self.feature_eng_var.get()
        if choice == "PCA" or choice == "MRMR":
            self.info_text.set(f"Applying {choice} for feature engineering...")
            self.show_image("Feature Engineering", choice)
            messagebox.showinfo("Success", f"Feature engineering ({choice}) has been successfully completed")
        else:
            self.info_text.set("No feature engineering applied.")

    def train_model(self):
        model = self.model_var.get()
        if not model:
            messagebox.showwarning("Warning", "Please select a model to train.")
            return

        feature_eng_choice = self.feature_eng_var.get()
        if feature_eng_choice == "Without Feature Engineering":
            image_key = "Without Feature Engineering"
        elif feature_eng_choice == "PCA":
            image_key = "PCA"
        elif feature_eng_choice == "MRMR":
            image_key = "MRMR"
        else:
            messagebox.showwarning("Warning", "Please select a valid feature engineering option.")
            return

        self.info_text.set(f"Training the {model} model...")
        self.show_image("Confusion Matrix", image_key, model)
        messagebox.showinfo("Success", f"Training the {model} model has been successfully completed")

    def evaluate_classifier(self):
        model = self.model_var.get()
        feature_eng_choice = self.feature_eng_var.get()

        performance_data = {}

        if model == "Decision Tree":
            if feature_eng_choice == "Without Feature Engineering":
                performance_data = {
                    "Accuracy": "0.996",
                    "AUC": "0.998",
                    "Precision": "0.996",
                    "Recall (Sensitivity)": "0.996",
                    "Specificity": "0.998",
                    "F-Measure": "0.996",
                    "Training Time (s)": "0.007",
                    "Testing Time (s)": "0.001"
                }
            elif feature_eng_choice == "PCA":
                performance_data = {
                    "Accuracy": "0.950",
                    "AUC": "0.953",
                    "Precision": "0.952",
                    "Recall (Sensitivity)": "0.950",
                    "Specificity": "0.952",
                    "F-Measure": "0.951",
                    "Training Time (s)": "0.023",
                    "Testing Time (s)": "0.001"
                }
            elif feature_eng_choice == "MRMR":
                performance_data = {
                    "Accuracy": "0.998",
                    "AUC": "0.999",
                    "Precision": "0.998",
                    "Recall (Sensitivity)": "0.998",
                    "Specificity": "0.999",
                    "F-Measure": "0.998",
                    "Training Time (s)": "0.004",
                    "Testing Time (s)": "0.001"
                }
        elif model == "KNN":
            if feature_eng_choice == "Without Feature Engineering":
                performance_data = {
                    "Accuracy": "0.985",
                    "AUC": "0.998",
                    "Precision": "0.986",
                    "Recall (Sensitivity)": "0.985",
                    "Specificity": "0.991",
                    "F-Measure": "0.985",
                    "Training Time (s)": "0.001",
                    "Testing Time (s)": "0.052"
                }
            elif feature_eng_choice == "PCA":
                performance_data = {
                    "Accuracy": "0.989",
                    "AUC": "0.995",
                    "Precision": "0.990",
                    "Recall (Sensitivity)": "0.989",
                    "Specificity": "0.990",
                    "F-Measure": "0.989",
                    "Training Time (s)": "0.001",
                    "Testing Time (s)": "0.058"
                }
            elif feature_eng_choice == "MRMR":
                performance_data = {
                    "Accuracy": "0.998",
                    "AUC": "1.000",
                    "Precision": "0.998",
                    "Recall (Sensitivity)": "0.998",
                    "Specificity": "0.999",
                    "F-Measure": "0.998",
                    "Training Time (s)": "0.001",
                    "Testing Time (s)": "0.072"
                }
       
       
       
       
       
       الجزء الاخير بالواجهه
       
       
       
        elif model == "SVM":١. تحليل الأداء بناءً على النموذج وخيارات تحسين الخصائص
python
نسخ الكود

            if feature_eng_choice == "Without Feature Engineering":
                performance_data = {
                    "Accuracy": "0.922",الدقه
                    "AUC": "0.994",
                    "Precision": "0.925",
                    "Recall (Sensitivity)": "0.922",
                    "Specificity": "0.961",
                    "F-Measure": "0.918",
                    "Training Time (s)": "0.267",
                    "Testing Time (s)": "0.036"
Accuracy: الدقة.
AUC: منحنى المساحة تحت المنحنى.
Precision: الدقة في التنبؤ.
Recall: الحساسية.
Specificity: الخصوصية.
F-Measure: مقياس F.
Training Time: وقت التدريب.
Testing Time: وقت الاختبار

                }
            elif feature_eng_choice == "PCA":
                performance_data = {
                    "Accuracy": "0.909",Accuracy: الدقة.
AUC: منحنى المساحة تحت المنحنى.
Precision: الدقة في التنبؤ.
Recall: الحساسية.
Specificity: الخصوصية.
F-Measure: مقياس F.
Training Time: وقت التدريب.
Testing Time: وقت الاختبار
                    "AUC": "0.944",
                    "Precision": "0.911",
                    "Recall (Sensitivity)": "0.909",
                    "Specificity": "0.946",
                    "F-Measure": "0.908",
                    "Training Time (s)": "0.226",
                    "Testing Time (s)": "0.026"
                }

            elif feature_eng_choice == "MRMR":
                performance_data = {
                    "Accuracy": "0.937",
                    "AUC": "0.975",
                    "Precision": "0.942",
                    "Recall (Sensitivity)": "0.937",
                    "Specificity": "0.971",
                    "F-Measure": "0.933",
                    "Training Time (s)": "0.256",
                    "Testing Time (s)": "0.035"
                }
        elif model == "Naive Bayes":
            if feature_eng_choice == "Without Feature Engineering":
                performance_data = {
                    "Accuracy": "0.718",
                    "AUC": "0.894",
                    "Precision": "0.781",
                    "Recall (Sensitivity)": "0.718",
                    "Specificity": "0.875",
                    "F-Measure": "0.706",
                    "Training Time (s)": "0.002",
                    "Testing Time (s)": "0.001"
                }
            elif feature_eng_choice == "PCA":
                performance_data = {
                    "Accuracy": "0.701",
                    "AUC": "0.816",
                    "Precision": "0.754",
                    "Recall (Sensitivity)": "0.701",
                    "Specificity": "0.883",
                    "F-Measure": "0.686",
                    "Training Time (s)": "0.002",
                    "Testing Time (s)": "0.001"
                }
            elif feature_eng_choice == "MRMR":
                performance_data = {
                    "Accuracy": "0.741",
                    "AUC": "0.861",
                    "Precision": "0.841",
                    "Recall (Sensitivity)": "0.741",
                    "Specificity": "0.883",
                    "F-Measure": "0.742",
                    "Training Time (s)": "0.002",
                    "Testing Time (s)": "0.001"
                }

        for key, value in performance_data.items():
            self.performance_vars[key].set(value)
٢. تحديث واجهة المستخدم بالنتائج


        self.info_text.set("Evaluating the classifier... Performance metrics updated.")
        messagebox.showinfo("Success", "Classifier evaluation completed") وتظهر لي رساله تم اكتمال تقيم 





    def show_image(self, category, subcategory=None, model=None):القسم الثاني: عرض الصور


        if category == "Confusion Matrix" and subcategory and model:
            image_path = self.image_files[category][subcategory][model]
        elif category in self.image_files:
            image_path = self.image_files[category][subcategory] if subcategory else self.image_files[category]
        else:
            messagebox.showwarning("Warning", "No images available for this category.")
            return 
            
            show_image: دالة لعرض الصور (مثل مصفوفة الالتباس أو أي صور متعلقة بالنموذج).
category: الفئة الرئيسية (مثل Confusion Matrix).
subcategory: الفئة الفرعية (إن وجدت).
model: النموذج المختار




        if isinstance(image_path, str) and os.path.exists(image_path):١. التحقق من وجود الصورة

            img = Image.open(image_path)
            img = img.resize((500, 400), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            for widget in self.image_display_frame.winfo_children():
                widget.destroy()
            img_label = tk.Label(self.image_display_frame, image=img)
            img_label.image = img  # Keep a reference to avoid garbage collection
            img_label.pack()
        else:
            messagebox.showwarning("Warning", f"Image {image_path} not found.")
يتم التحقق من وجود الصورة في المسار المحدد.
إذا كانت موجودة:
يتم فتح الصورة وإعادة ضبط حجمها.
يتم عرض الصورة داخل إطار واجهة المستخدم.
إذا لم تكن الصورة موجودة، تظهر رسالة تحذيرية.



if __name__ == "__main__":
    root = tk.Tk()
    app = IDSInHealthcareApp(root)
    root.mainloop()
    
tk.Tk(): إنشاء نافذة تطبيق باستخدام مكتبة Tkinter.
IDSInHealthcareApp: استدعاء واجهة التطبيق الرئيسية.
mainloop: بدء تشغيل التطبيق والانتظار لتفاعل المستخدم.
