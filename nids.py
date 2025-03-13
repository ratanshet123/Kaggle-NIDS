import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE  # For handling dataset imbalance

# Load dataset
df = pd.read_csv('KDDTrain+.txt', header=None)

# Define column names
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", 
           "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", 
           "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", 
           "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
           "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
           "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
           "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
           "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
           "dst_host_srv_rerror_rate", "label", "difficulty"]
df.columns = columns

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Drop 'difficulty' column
df.drop(columns=['difficulty'], inplace=True)

# Convert labels to binary (normal = 0, attack = 1)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Check dataset balance
print("Label distribution before balancing:\n", df['label'].value_counts())

# Split features and labels
X = df.drop(columns=['label'])
y = df['label']

# Handle dataset imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize the data
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model with class weight balancing
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.6f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix Heatmap')
plt.savefig('./confusion_matrix_heatmap.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('./roc_curve.png')
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('./precision_recall_curve.png')
plt.close()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(df.columns[:-1])[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.savefig('./feature_importance.png')
plt.close()

# Save model and scaler
joblib.dump(model, "ids_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")

# Test a known attack sample
attack_sample = np.array([[9000, 0, 1, 2, 20000, 70000, 1, 15, 10, 100, 100, 0, 50, 1, 1, 40, 20, 10, 20, 0, 1, 800, 400, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 255, 255, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,0]])  # Added missing feature

attack_sample = scaler.transform(attack_sample)
print("Test Attack Prediction:", model.predict(attack_sample))
