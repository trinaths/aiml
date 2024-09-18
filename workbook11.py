# Step 1: Import Libraries/Data Set
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, RocCurveDisplay

# Load the Titanic dataset
url = "data.csv"
data = pd.read_csv(url)

# Step 2: Data Visualization and Augmentation
# EDA Graphs
# 1. Countplot of survival
plt.figure(figsize=(10, 5))
sns.countplot(x='Survived', data=data)
plt.title('Count of Survival')
plt.show()



# 2. Age distribution of passengers
plt.figure(figsize=(10, 5))
sns.histplot(data['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

# Step 3: Prepare Data for Model
# Dropping unnecessary columns and handling missing values
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Splitting data into features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print shapes of train and test data
print("Training data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

# OUTPUT
# Training data shape: (712, 6)
# Testing data shape: (179, 6)

# Step 4: Naïve Bayes Model Building
# Building Naïve Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Model summary
print("Naïve Bayes Model Trained")

# Step 5: SVM Model Building
# Building SVM model
svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)

# Model summary
print("SVM Model Trained")

# Step 6: Model Evaluation
# Naïve Bayes Evaluation
nb_predictions = nb_model.predict(X_test_scaled)
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_confusion = confusion_matrix(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)
nb_auc = roc_auc_score(y_test, nb_model.predict_proba(X_test_scaled)[:, 1])

print("Naïve Bayes Accuracy:", nb_accuracy)
print("Naïve Bayes Confusion Matrix:\n", nb_confusion)
print("Naïve Bayes F1 Score:", nb_f1)
print("Naïve Bayes AUC:", nb_auc)

# OUTPUT
# Naïve Bayes Accuracy: 0.770949720670391
# Naïve Bayes Confusion Matrix:
#  [[85 20]
#  [21 53]]
# Naïve Bayes F1 Score: 0.7210884353741497
# Naïve Bayes AUC: 0.8616473616473617


# Plot AUC-ROC for Naïve Bayes
RocCurveDisplay.from_estimator(nb_model, X_test_scaled, y_test)
plt.title('Naïve Bayes AUC-ROC')
plt.show()

# SVM Evaluation
svm_predictions = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_confusion = confusion_matrix(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)
svm_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test_scaled)[:, 1])

print("SVM Accuracy:", svm_accuracy)
print("SVM Confusion Matrix:\n", svm_confusion)
print("SVM F1 Score:", svm_f1)
print("SVM AUC:", svm_auc)

## OUTPUT
# SVM Accuracy: 0.8100558659217877
# SVM Confusion Matrix:
# [[92 13]
# [21 53]]
# SVM F1 Score: 0.7571428571428571
# SVM AUC: 0.8445302445302446


# Plot AUC-ROC for SVM
RocCurveDisplay.from_estimator(svm_model, X_test_scaled, y_test)
plt.title('SVM AUC-ROC')
plt.show()

# Step 7: Compare Performances
print("\nComparison of Naïve Bayes and SVM:")
print(f"Naïve Bayes Accuracy: {nb_accuracy}, SVM Accuracy: {svm_accuracy}")
print(f"Naïve Bayes F1 Score: {nb_f1}, SVM F1 Score: {svm_f1}")
print(f"Naïve Bayes AUC: {nb_auc}, SVM AUC: {svm_auc}")

# OUTPUT

# Comparison of Naïve Bayes and SVM:
# Naïve Bayes Accuracy: 0.770949720670391, SVM Accuracy: 0.8100558659217877
# Naïve Bayes F1 Score: 0.7210884353741497, SVM F1 Score: 0.7571428571428571
# Naïve Bayes AUC: 0.8616473616473617, SVM AUC: 0.8445302445302446