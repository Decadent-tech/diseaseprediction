import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv("dataset\improved_disease_dataset.csv")
print(data.head())

# Data Preprocessing
print("Data shape:", data.shape)
print("Data columns:", data.columns)
print("Data information:",data.info())
print("Data description:\n", data.describe(include='all'))


# Visualize the data
plt.figure(figsize=(10, 6))
for column in data.columns:
    if data[column].dtype == 'object':
        plt.figure(figsize=(10, 6))
        sns.countplot(data[column])
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
        plt.savefig(f'visualize\distribution_{column}.png')
        #plt.show()
    if data[column].dtype in ['int64', 'float64']:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(f'visualize\histogram_{column}.png')
        #plt.show()

# Check duplicates 
print("Number of duplicate rows:", data.duplicated().sum())
#each customer represnt a row so no duplicates is considered 

print(data["disease"].nunique())

#label encoding on disease column
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

plt.figure(figsize=(18, 8))
sns.countplot(x=y)
plt.xlabel("Disease Class")
plt.title("Disease Class Distribution Before Resampling")
plt.xticks(rotation=90)
plt.savefig('visualize/disease_class_distribution_before_resampling.png')
#plt.show()

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("Resampled Class Distribution:\n", pd.Series(y_resampled).value_counts())


#Cross-Validation with Stratified K-Fold
from sklearn.model_selection import StratifiedKFold 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Model Training and Evaluation
if 'gender' in X_resampled.columns:
    le = LabelEncoder()
    X_resampled['gender'] = le.fit_transform(X_resampled['gender'])

X_resampled = X_resampled.fillna(0)

if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()

models = {
    "SVM": SVC(kernel='linear', random_state=42),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

cv_scoring = 'accuracy'  # you can also use 'f1_weighted', 'roc_auc_ovr' for multi-class
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    try:
        scores = cross_val_score(
            model,
            X_resampled,
            y_resampled,
            cv=stratified_kfold,
            scoring=cv_scoring,
            n_jobs=-1,
            error_score='raise' 
        )
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}")
    except Exception as e:
        print("=" * 50)
        print(f"Model: {model_name} failed with error:")
        print(e)

#Training Individual Models and Generating Confusion Matrices

svm_model = SVC()
svm_model.fit(X_resampled, y_resampled)
svm_preds = svm_model.predict(X_resampled)

cf_matrix_svm = confusion_matrix(y_resampled, svm_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_svm, annot=True, fmt="d")
plt.title("Confusion Matrix for SVM Classifier")
plt.savefig('visualize/confusion_matrix_svm.png')
plt.show()

print(f"SVM Accuracy: {accuracy_score(y_resampled, svm_preds) * 100:.2f}%")
#The matrix shows good accuracy with most values along the diagonal meaning the SVM model predicted the correct class most of the time.

nb_model = GaussianNB()
nb_model.fit(X_resampled, y_resampled)
nb_preds = nb_model.predict(X_resampled)

cf_matrix_nb = confusion_matrix(y_resampled, nb_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_nb, annot=True, fmt="d")
plt.title("Confusion Matrix for Naive Bayes Classifier")
plt.savefig('visualize/confusion_matrix_nb.png')
plt.show()

print(f"Naive Bayes Accuracy: {accuracy_score(y_resampled, nb_preds) * 100:.2f}%")
#This matrix shows many off-diagonal values meaning the Naive Bayes model made more errors compared to the SVM. 
# The predictions are less accurate and more spread out across incorrect classes.

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)
rf_preds = rf_model.predict(X_resampled)

cf_matrix_rf = confusion_matrix(y_resampled, rf_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_rf, annot=True, fmt="d")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.savefig('visualize/confusion_matrix_rf.png')
plt.show()

print(f"Random Forest Accuracy: {accuracy_score(y_resampled, rf_preds) * 100:.2f}%")

#This confusion matrix shows strong performance with most predictions correctly placed along the diagonal. 
# It has fewer misclassifications than Naive Bayes and is comparable or slightly better than SVM.

#combining the prediction for robustness this reduces teh variances 
from statistics import mode

final_preds = [mode([i, j, k]) for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

cf_matrix_combined = confusion_matrix(y_resampled, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_combined, annot=True, fmt="d")
plt.title("Confusion Matrix for Combined Model")
plt.savefig('visualize/confusion_matrix_combined.png')
plt.show()

print(f"Combined Model Accuracy: {accuracy_score(y_resampled, final_preds) * 100:.2f}%")

#Combined Model Accuracy: 60.64%

