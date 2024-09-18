# Import Libraries/Data set (1 point)
# Import the required libraries and the dataset
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

def evaluate_model(y_true, y_pred, y_probs, model_name):
    print(f"{model_name} Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"{model_name} Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"{model_name} F1 Score: {f1_score(y_true, y_pred)}")
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred), f1_score(y_true, y_pred), roc_auc


# Data Visualisation and Augmentation (0.5*6 = 3 points)
# Plot at least two EDA graphs (use matplotlib/seaborn/any other library)
# Prepare data to be able to build a classification model
# Bring the train and test data in the required format
# Perform missing values check
# Perform scaling of data
# Print the shapes of train and test data

plt.figure(figsize=(10, 5))
sns.countplot(x='Survived', data=data)
plt.title('Count of Survival')
plt.show()

# 2. Age distribution of passengers
plt.figure(figsize=(10, 5))
sns.histplot(data['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

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

# Naïve Bayes Model Building (0.5*3 = 1.5 points)
# Build a Naïve bayes classification model
# Train the model on the train data set 
# Print the model summary
nb_model = GaussianNB()

# nb_model.fit(X_train_scaled, y_train) trains the 
# Naïve Bayes model on the scaled training data (X_train_scaled) 
# and the corresponding target values (y_train)
nb_model.fit(X_train_scaled, y_train)

# Model summary
print("Naïve Bayes Model Trained")

# SVM Model Building (0.5*3 = 1.5 points)
# Build an SVM classification model
# Train the model on the train data set 
# Print the model summary

svm_model = SVC(probability=True)

#svm_model.fit(X_train_scaled, y_train) trains the 
# SVM model similarly. The parameter probability=True 
# is set to ensure the model can provide probability 
# estimates needed for ROC curve calculations.
svm_model.fit(X_train_scaled, y_train)

# Model summary
print("SVM Model Trained")

# Model Evaluation (1 + 1 = 2 points)
# Check the Naïve bayes model’s performance by printing accuracy, confusion matrix, F1 score and AUC-ROC curve
# Check the SVM model’s performance by printing accuracy, confusion matrix, F1 score and AUC-ROC curve 

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

# 5. Model Evaluation

# Naïve Bayes Predictions
nb_predictions = nb_model.predict(X_test_scaled)
nb_probs = nb_model.predict_proba(X_test_scaled)[:, 1]

# SVM Predictions
svm_predictions = svm_model.predict(X_test_scaled)
svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]

# Comparing performances
print("\nModel Comparison:")
print(f"Naïve Bayes Accuracy: {nb_accuracy:.4f}, F1 Score: {nb_f1:.4f}, AUC: {nb_auc:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}, F1 Score: {svm_f1:.4f}, AUC: {svm_auc:.4f}")

if svm_accuracy > nb_accuracy:
    print("SVM performs better than Naïve Bayes in terms of accuracy.")
else:
    print("Naïve Bayes performs better than SVM in terms of accuracy.")

if svm_f1 > nb_f1:
    print("SVM has a better F1 Score than Naïve Bayes.")
else:
    print("Naïve Bayes has a better F1 Score than SVM.")

if svm_auc > nb_auc:
    print("SVM has a better AUC than Naïve Bayes.")
else:
    print("Naïve Bayes has a better AUC than SVM.")

# OUTPUT
# Model Comparison:
# Naïve Bayes Accuracy: 0.7709, F1 Score: 0.7211, AUC: 0.8616
# SVM Accuracy: 0.8101, F1 Score: 0.7571, AUC: 0.8445
# SVM performs better than Naïve Bayes in terms of accuracy.
# SVM has a better F1 Score than Naïve Bayes.
# Naïve Bayes has a better AUC than SVM.


