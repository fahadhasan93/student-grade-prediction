import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

# Load the dataset
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Green%20University%20Student%20Grade%20Prediction%20Dataset-jRvbhln0jqdm32et2PfosRnZSOrZa3.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nUnique values in categorical columns:")
categorical_cols = ['gender', 'parent_education', 'employment_status', 'online_resource_usage', 'final_grade']
for col in categorical_cols:
    if col in df.columns:
        print(f"{col}: {df[col].unique()}")

# Data cleaning and preprocessing
print("\n=== DATA CLEANING ===")

# Remove unnecessary columns
columns_to_remove = ['student_id', 'name', 'email', 'address']
df_clean = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

# Convert string numbers to numeric
numeric_cols = ['study_hours_per_week', 'library_visits_per_month', 'ct_average', 
                'midterm_score', 'final_score', 'total_mark']

for col in numeric_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Handle missing values
df_clean = df_clean.dropna()

print(f"Dataset shape after cleaning: {df_clean.shape}")

# Feature Engineering
print("\n=== FEATURE ENGINEERING ===")

# Create new features
if 'ct1_score' in df_clean.columns and 'ct2_score' in df_clean.columns:
    df_clean['ct_improvement'] = df_clean['ct2_score'] - df_clean['ct1_score']

if 'study_hours_per_week' in df_clean.columns:
    df_clean['study_intensity'] = pd.cut(df_clean['study_hours_per_week'], 
                                       bins=[0, 5, 10, 15, float('inf')], 
                                       labels=['Low', 'Medium', 'High', 'Very High'])

# Prepare features for modeling
X = df_clean.drop('final_grade', axis=1)
y = df_clean['final_grade']

# Encode categorical variables
label_encoders = {}
categorical_features = X.select_dtypes(include=['object']).columns

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Model Building
print("\n=== MODEL BUILDING ===")

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Model Evaluation
print("\n=== MODEL EVALUATION ===")

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save model artifacts for the web application
model_data = {
    'feature_names': X.columns.tolist(),
    'label_encoders': {k: v.classes_.tolist() for k, v in label_encoders.items()},
    'feature_importance': feature_importance.to_dict('records'),
    'accuracy': accuracy,
    'grade_classes': sorted(y.unique().tolist())
}

print("\nModel training completed successfully!")
print(f"Model artifacts saved for web application")
