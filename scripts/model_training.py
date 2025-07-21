import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import json

print("=== ADVANCED MODEL TRAINING ===")

# Load the dataset
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Green%20University%20Student%20Grade%20Prediction%20Dataset-jRvbhln0jqdm32et2PfosRnZSOrZa3.csv"
df = pd.read_csv(url)

print(f"Dataset loaded: {df.shape}")

# Advanced data preprocessing
def preprocess_data(df):
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
    
    # Feature engineering
    if 'ct1_score' in df_clean.columns and 'ct2_score' in df_clean.columns:
        df_clean['ct_improvement'] = df_clean['ct2_score'] - df_clean['ct1_score']
        df_clean['ct_consistency'] = abs(df_clean['ct2_score'] - df_clean['ct1_score'])
    
    if 'study_hours_per_week' in df_clean.columns:
        df_clean['study_category'] = pd.cut(df_clean['study_hours_per_week'], 
                                          bins=[0, 5, 10, 15, float('inf')], 
                                          labels=[0, 1, 2, 3])
    
    if 'family_income' in df_clean.columns:
        df_clean['income_category'] = pd.cut(df_clean['family_income'], 
                                           bins=[0, 500000, 1000000, 2000000, float('inf')], 
                                           labels=[0, 1, 2, 3])
    
    # Create academic performance score
    academic_cols = ['ct1_score', 'ct2_score', 'assignment_score', 'presentation_score']
    if all(col in df_clean.columns for col in academic_cols):
        df_clean['academic_performance'] = df_clean[academic_cols].mean(axis=1)
    
    return df_clean

df_processed = preprocess_data(df)
print(f"Data after preprocessing: {df_processed.shape}")

# Prepare features and target
X = df_processed.drop('final_grade', axis=1)
y = df_processed['final_grade']

print(f"Features: {X.columns.tolist()}")
print(f"Target classes: {sorted(y.unique())}")

# Encode categorical variables
label_encoders = {}
categorical_features = X.select_dtypes(include=['object']).columns

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Hyperparameter tuning
print("\n=== HYPERPARAMETER TUNING ===")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== MODEL EVALUATION ===")
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n=== FEATURE IMPORTANCE ===")
print(feature_importance.head(15))

# Grade distribution analysis
print(f"\n=== GRADE DISTRIBUTION ===")
grade_dist = y.value_counts().sort_index()
print("Training data grade distribution:")
print(grade_dist)

print("\nPredicted grade distribution:")
pred_dist = pd.Series(y_pred).value_counts().sort_index()
print(pred_dist)

# Model insights
print(f"\n=== MODEL INSIGHTS ===")
print(f"Total number of trees: {best_model.n_estimators}")
print(f"Maximum depth: {best_model.max_depth}")
print(f"Number of features used: {X.shape[1]}")
print(f"Most important feature: {feature_importance.iloc[0]['feature']}")

# Save model summary
model_summary = {
    'accuracy': accuracy,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'best_params': grid_search.best_params_,
    'feature_importance': feature_importance.to_dict('records'),
    'grade_distribution': grade_dist.to_dict(),
    'feature_names': X.columns.tolist(),
    'label_encoders': {k: v.classes_.tolist() for k, v in label_encoders.items()}
}

print("\n=== MODEL TRAINING COMPLETED ===")
print("Model summary saved for web application integration")
