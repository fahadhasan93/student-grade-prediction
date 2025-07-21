import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import pickle
import json
from collections import Counter

print("=== ADVANCED MODEL DEVELOPMENT ===")

# Load and analyze the dataset
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Green%20University%20Student%20Grade%20Prediction%20Dataset-jRvbhln0jqdm32et2PfosRnZSOrZa3.csv"
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nGrade distribution:")
grade_counts = df['final_grade'].value_counts().sort_index()
print(grade_counts)
print(f"Grade percentages:")
print((grade_counts / len(df) * 100).round(2))

# Advanced preprocessing pipeline
def advanced_preprocessing(df):
    df_clean = df.copy()
    
    # Remove unnecessary columns
    columns_to_remove = ['student_id', 'name', 'email', 'address']
    df_clean = df_clean.drop(columns=[col for col in columns_to_remove if col in df_clean.columns])
    
    # Convert string numbers to numeric with better error handling
    numeric_cols = ['study_hours_per_week', 'library_visits_per_month', 'ct_average', 
                    'midterm_score', 'final_score', 'total_mark']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Handle missing values intelligently
    print(f"\nMissing values before handling:")
    print(df_clean.isnull().sum())
    
    # Fill numeric columns with median
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'final_grade' and df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Advanced feature engineering
    print(f"\n=== FEATURE ENGINEERING ===")
    
    # 1. CT Performance Analysis
    if 'ct1_score' in df_clean.columns and 'ct2_score' in df_clean.columns:
        df_clean['ct_improvement'] = df_clean['ct2_score'] - df_clean['ct1_score']
        df_clean['ct_average_manual'] = (df_clean['ct1_score'] + df_clean['ct2_score']) / 2
        df_clean['ct_consistency'] = 1 / (1 + abs(df_clean['ct2_score'] - df_clean['ct1_score']))
        df_clean['ct_trend'] = np.where(df_clean['ct_improvement'] > 0, 1, 
                                       np.where(df_clean['ct_improvement'] < 0, -1, 0))
    
    # 2. Academic Performance Score
    academic_components = []
    weights = {}
    
    if 'ct1_score' in df_clean.columns:
        academic_components.append('ct1_score')
        weights['ct1_score'] = 0.1
    if 'ct2_score' in df_clean.columns:
        academic_components.append('ct2_score')
        weights['ct2_score'] = 0.1
    if 'assignment_score' in df_clean.columns:
        academic_components.append('assignment_score')
        weights['assignment_score'] = 0.2
    if 'presentation_score' in df_clean.columns:
        academic_components.append('presentation_score')
        weights['presentation_score'] = 0.15
    if 'midterm_score' in df_clean.columns:
        academic_components.append('midterm_score')
        weights['midterm_score'] = 0.25
    if 'final_score' in df_clean.columns:
        academic_components.append('final_score')
        weights['final_score'] = 0.2
    
    if academic_components:
        # Weighted academic score
        df_clean['weighted_academic_score'] = 0
        for component in academic_components:
            df_clean['weighted_academic_score'] += df_clean[component] * weights.get(component, 0.1)
        
        # Academic consistency
        df_clean['academic_std'] = df_clean[academic_components].std(axis=1)
        df_clean['academic_range'] = df_clean[academic_components].max(axis=1) - df_clean[academic_components].min(axis=1)
    
    # 3. Study Behavior Analysis
    if 'study_hours_per_week' in df_clean.columns:
        df_clean['study_intensity'] = pd.cut(df_clean['study_hours_per_week'], 
                                           bins=[0, 5, 10, 15, 20, float('inf')], 
                                           labels=[1, 2, 3, 4, 5])
        df_clean['study_intensity'] = df_clean['study_intensity'].astype(float)
        
        # Study efficiency
        if 'total_mark' in df_clean.columns:
            df_clean['study_efficiency'] = df_clean['total_mark'] / (df_clean['study_hours_per_week'] + 1)
    
    # 4. Resource Utilization Score
    resource_score = 0
    if 'library_visits_per_month' in df_clean.columns:
        df_clean['library_usage'] = pd.cut(df_clean['library_visits_per_month'], 
                                         bins=[0, 2, 5, 10, float('inf')], 
                                         labels=[1, 2, 3, 4])
        df_clean['library_usage'] = df_clean['library_usage'].astype(float)
        resource_score += df_clean['library_usage']
    
    if 'online_resource_usage' in df_clean.columns:
        online_mapping = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5}
        df_clean['online_usage_score'] = df_clean['online_resource_usage'].map(online_mapping)
        resource_score += df_clean['online_usage_score']
    
    df_clean['total_resource_utilization'] = resource_score
    
    # 5. Socioeconomic Impact
    if 'family_income' in df_clean.columns:
        df_clean['income_log'] = np.log1p(df_clean['family_income'])
        df_clean['income_category'] = pd.cut(df_clean['family_income'], 
                                           bins=[0, 300000, 600000, 1000000, 2000000, float('inf')], 
                                           labels=[1, 2, 3, 4, 5])
        df_clean['income_category'] = df_clean['income_category'].astype(float)
    
    # 6. Parent Education Impact
    if 'parent_education' in df_clean.columns:
        education_mapping = {'High School': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
        df_clean['parent_education_score'] = df_clean['parent_education'].map(education_mapping)
    
    # 7. Age-based features
    if 'age' in df_clean.columns:
        df_clean['age_category'] = pd.cut(df_clean['age'], 
                                        bins=[0, 20, 22, 25, float('inf')], 
                                        labels=[1, 2, 3, 4])
        df_clean['age_category'] = df_clean['age_category'].astype(float)
    
    # 8. Performance ratios
    if 'midterm_score' in df_clean.columns and 'final_score' in df_clean.columns:
        df_clean['final_to_midterm_ratio'] = df_clean['final_score'] / (df_clean['midterm_score'] + 1)
    
    print(f"Features after engineering: {df_clean.shape[1]}")
    print(f"New features created: {df_clean.shape[1] - df.shape[1] + len(columns_to_remove)}")
    
    return df_clean

# Apply preprocessing
df_processed = advanced_preprocessing(df)

# Remove rows with missing target
df_processed = df_processed.dropna(subset=['final_grade'])
print(f"Final dataset shape: {df_processed.shape}")

# Prepare features and target
X = df_processed.drop('final_grade', axis=1)
y = df_processed['final_grade']

print(f"Features: {X.columns.tolist()}")
print(f"Target classes: {sorted(y.unique())}")

# Encode categorical variables
label_encoders = {}
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Categorical features to encode: {categorical_features.tolist()}")

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Feature selection
print(f"\n=== FEATURE SELECTION ===")
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X, y)
feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("Top 15 features by F-score:")
print(feature_scores.head(15))

# Select top features
top_features = feature_scores.head(20)['feature'].tolist()
X_top = X[top_features]

print(f"Using top {len(top_features)} features for modeling")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_top, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))
print(f"Class weights: {class_weight_dict}")

# Model 1: Advanced Random Forest
print(f"\n=== RANDOM FOREST MODEL ===")
rf_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

print(f"Best RF parameters: {rf_grid.best_params_}")
print(f"Best RF CV score: {rf_grid.best_score_:.4f}")

best_rf = rf_grid.best_estimator_
rf_pred = best_rf.predict(X_test)
rf_accuracy = balanced_accuracy_score(y_test, rf_pred)

print(f"RF Test Balanced Accuracy: {rf_accuracy:.4f}")

# Model 2: Gradient Boosting
print(f"\n=== GRADIENT BOOSTING MODEL ===")
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
gb_grid.fit(X_train, y_train)

print(f"Best GB parameters: {gb_grid.best_params_}")
print(f"Best GB CV score: {gb_grid.best_score_:.4f}")

best_gb = gb_grid.best_estimator_
gb_pred = best_gb.predict(X_test)
gb_accuracy = balanced_accuracy_score(y_test, gb_pred)

print(f"GB Test Balanced Accuracy: {gb_accuracy:.4f}")

# Choose best model
if rf_accuracy > gb_accuracy:
    best_model = best_rf
    best_pred = rf_pred
    best_accuracy = rf_accuracy
    model_name = "Random Forest"
else:
    best_model = best_gb
    best_pred = gb_pred
    best_accuracy = gb_accuracy
    model_name = "Gradient Boosting"

print(f"\n=== BEST MODEL: {model_name} ===")
print(f"Balanced Accuracy: {best_accuracy:.4f}")
print(f"Standard Accuracy: {accuracy_score(y_test, best_pred):.4f}")

# Detailed evaluation
print(f"\nClassification Report:")
print(classification_report(y_test, best_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Per-class metrics
print(f"\nPer-class Performance:")
for i, grade in enumerate(sorted(y.unique())):
    if i < len(cm):
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{grade}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

# Cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='balanced_accuracy')
print(f"\nCross-validation Balanced Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save model artifacts
model_artifacts = {
    'model_type': model_name,
    'feature_names': X_train.columns.tolist(),
    'label_encoders': {k: v.classes_.tolist() for k, v in label_encoders.items()},
    'scaler_mean': scaler.center_.tolist() if hasattr(scaler, 'center_') else None,
    'scaler_scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
    'class_weights': class_weight_dict,
    'feature_importance': feature_importance.to_dict('records') if hasattr(best_model, 'feature_importances_') else None,
    'accuracy_metrics': {
        'balanced_accuracy': best_accuracy,
        'standard_accuracy': accuracy_score(y_test, best_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    },
    'grade_mapping': {grade: i for i, grade in enumerate(sorted(y.unique()))},
    'top_features': top_features
}

# Save the trained model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model_artifacts.json', 'w') as f:
    json.dump(model_artifacts, f, indent=2)

print(f"\n=== MODEL TRAINING COMPLETED ===")
print(f"Best model: {model_name}")
print(f"Balanced Accuracy: {best_accuracy:.4f}")
print(f"Model and artifacts saved successfully")
print(f"Ready for integration with web application")
