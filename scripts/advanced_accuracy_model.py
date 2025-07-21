import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=== ADVANCED HIGH-ACCURACY MODEL DEVELOPMENT ===")

# Load the dataset
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Green%20University%20Student%20Grade%20Prediction%20Dataset-jRvbhln0jqdm32et2PfosRnZSOrZa3.csv"
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print(f"Grade distribution:")
print(df['final_grade'].value_counts().sort_index())

# Advanced feature engineering for maximum accuracy
def advanced_feature_engineering(df):
    df_enhanced = df.copy()
    
    # Remove unnecessary columns
    columns_to_remove = ['student_id', 'name', 'email', 'address']
    df_enhanced = df_enhanced.drop(columns=[col for col in columns_to_remove if col in df_enhanced.columns])
    
    # Convert all numeric columns properly
    numeric_cols = ['age', 'family_income', 'study_hours_per_week', 'library_visits_per_month', 
                    'ct1_score', 'ct2_score', 'assignment_score', 'presentation_score', 
                    'midterm_score', 'final_score', 'total_mark', 'ct_average']
    
    for col in numeric_cols:
        if col in df_enhanced.columns:
            df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors='coerce')
    
    # Handle missing values with advanced imputation
    for col in df_enhanced.columns:
        if col != 'final_grade':
            if df_enhanced[col].dtype in ['float64', 'int64']:
                # Use median for numeric columns
                df_enhanced[col].fillna(df_enhanced[col].median(), inplace=True)
            else:
                # Use mode for categorical columns
                df_enhanced[col].fillna(df_enhanced[col].mode()[0], inplace=True)
    
    # === ADVANCED FEATURE ENGINEERING ===
    
    # 1. Academic Performance Features
    if all(col in df_enhanced.columns for col in ['ct1_score', 'ct2_score']):
        df_enhanced['ct_improvement'] = df_enhanced['ct2_score'] - df_enhanced['ct1_score']
        df_enhanced['ct_improvement_rate'] = df_enhanced['ct_improvement'] / (df_enhanced['ct1_score'] + 1)
        df_enhanced['ct_consistency'] = 1 / (1 + abs(df_enhanced['ct2_score'] - df_enhanced['ct1_score']))
        df_enhanced['ct_average_manual'] = (df_enhanced['ct1_score'] + df_enhanced['ct2_score']) / 2
        df_enhanced['ct_momentum'] = np.where(df_enhanced['ct_improvement'] > 0, 1, 
                                            np.where(df_enhanced['ct_improvement'] < 0, -1, 0))
    
    # 2. Comprehensive Academic Score
    academic_components = []
    weights = {}
    
    if 'ct1_score' in df_enhanced.columns:
        academic_components.append('ct1_score')
        weights['ct1_score'] = 0.1
    if 'ct2_score' in df_enhanced.columns:
        academic_components.append('ct2_score')
        weights['ct2_score'] = 0.1
    if 'assignment_score' in df_enhanced.columns:
        academic_components.append('assignment_score')
        weights['assignment_score'] = 0.2
    if 'presentation_score' in df_enhanced.columns:
        academic_components.append('presentation_score')
        weights['presentation_score'] = 0.15
    if 'midterm_score' in df_enhanced.columns:
        academic_components.append('midterm_score')
        weights['midterm_score'] = 0.25
    if 'final_score' in df_enhanced.columns:
        academic_components.append('final_score')
        weights['final_score'] = 0.2
    
    if academic_components:
        # Normalize each component to 0-100 scale
        normalized_components = []
        max_scores = {'ct1_score': 10, 'ct2_score': 10, 'assignment_score': 20, 
                     'presentation_score': 15, 'midterm_score': 30, 'final_score': 40}
        
        for component in academic_components:
            max_score = max_scores.get(component, df_enhanced[component].max())
            normalized = (df_enhanced[component] / max_score) * 100
            normalized_components.append(normalized)
            df_enhanced[f'{component}_normalized'] = normalized
        
        # Weighted academic performance
        df_enhanced['weighted_academic_score'] = 0
        for i, component in enumerate(academic_components):
            df_enhanced['weighted_academic_score'] += normalized_components[i] * weights[component]
        
        # Academic statistics
        df_enhanced['academic_mean'] = np.mean(normalized_components, axis=0)
        df_enhanced['academic_std'] = np.std(normalized_components, axis=0)
        df_enhanced['academic_min'] = np.min(normalized_components, axis=0)
        df_enhanced['academic_max'] = np.max(normalized_components, axis=0)
        df_enhanced['academic_range'] = df_enhanced['academic_max'] - df_enhanced['academic_min']
        df_enhanced['academic_cv'] = df_enhanced['academic_std'] / (df_enhanced['academic_mean'] + 1)
    
    # 3. Study Behavior Analysis
    if 'study_hours_per_week' in df_enhanced.columns:
        df_enhanced['study_intensity'] = pd.cut(df_enhanced['study_hours_per_week'], 
                                              bins=[0, 5, 10, 15, 20, 25, float('inf')], 
                                              labels=[1, 2, 3, 4, 5, 6]).astype(float)
        
        # Study efficiency metrics
        if 'total_mark' in df_enhanced.columns:
            df_enhanced['study_efficiency'] = df_enhanced['total_mark'] / (df_enhanced['study_hours_per_week'] + 1)
        
        if 'weighted_academic_score' in df_enhanced.columns:
            df_enhanced['study_roi'] = df_enhanced['weighted_academic_score'] / (df_enhanced['study_hours_per_week'] + 1)
    
    # 4. Resource Utilization Score
    resource_score = 0
    
    if 'library_visits_per_month' in df_enhanced.columns:
        df_enhanced['library_intensity'] = pd.cut(df_enhanced['library_visits_per_month'], 
                                                bins=[0, 2, 5, 10, 15, float('inf')], 
                                                labels=[1, 2, 3, 4, 5]).astype(float)
        resource_score += df_enhanced['library_intensity']
    
    if 'online_resource_usage' in df_enhanced.columns:
        online_mapping = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5}
        df_enhanced['online_usage_score'] = df_enhanced['online_resource_usage'].map(online_mapping)
        resource_score += df_enhanced['online_usage_score']
    
    df_enhanced['total_resource_utilization'] = resource_score
    
    # 5. Socioeconomic Impact Analysis
    if 'family_income' in df_enhanced.columns:
        df_enhanced['income_log'] = np.log1p(df_enhanced['family_income'])
        df_enhanced['income_category'] = pd.cut(df_enhanced['family_income'], 
                                              bins=[0, 200000, 500000, 1000000, 2000000, float('inf')], 
                                              labels=[1, 2, 3, 4, 5]).astype(float)
        df_enhanced['income_normalized'] = MinMaxScaler().fit_transform(df_enhanced[['family_income']])[:, 0]
    
    # 6. Parent Education Impact
    if 'parent_education' in df_enhanced.columns:
        education_mapping = {'High School': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
        df_enhanced['parent_education_score'] = df_enhanced['parent_education'].map(education_mapping)
    
    # 7. Age-based features
    if 'age' in df_enhanced.columns:
        df_enhanced['age_category'] = pd.cut(df_enhanced['age'], 
                                           bins=[0, 19, 21, 23, 25, float('inf')], 
                                           labels=[1, 2, 3, 4, 5]).astype(float)
        df_enhanced['age_normalized'] = MinMaxScaler().fit_transform(df_enhanced[['age']])[:, 0]
    
    # 8. Employment Impact
    if 'employment_status' in df_enhanced.columns:
        employment_mapping = {'Unemployed': 3, 'Part-time': 2, 'Full-time': 1}  # Unemployed might be better for studies
        df_enhanced['employment_score'] = df_enhanced['employment_status'].map(employment_mapping)
    
    # 9. Gender Impact
    if 'gender' in df_enhanced.columns:
        df_enhanced['gender_encoded'] = LabelEncoder().fit_transform(df_enhanced['gender'])
    
    # 10. Interaction Features (Most Important for Accuracy)
    if all(col in df_enhanced.columns for col in ['study_hours_per_week', 'library_visits_per_month']):
        df_enhanced['study_library_interaction'] = df_enhanced['study_hours_per_week'] * df_enhanced['library_visits_per_month']
    
    if all(col in df_enhanced.columns for col in ['weighted_academic_score', 'study_hours_per_week']):
        df_enhanced['academic_study_interaction'] = df_enhanced['weighted_academic_score'] * df_enhanced['study_hours_per_week']
    
    if all(col in df_enhanced.columns for col in ['midterm_score', 'final_score']):
        df_enhanced['midterm_final_ratio'] = df_enhanced['final_score'] / (df_enhanced['midterm_score'] + 1)
        df_enhanced['midterm_final_sum'] = df_enhanced['midterm_score'] + df_enhanced['final_score']
    
    # 11. Performance Ratios
    if all(col in df_enhanced.columns for col in ['ct_average_manual', 'midterm_score']):
        df_enhanced['ct_to_midterm_ratio'] = df_enhanced['ct_average_manual'] / (df_enhanced['midterm_score'] + 1)
    
    # 12. Risk Factors
    df_enhanced['low_study_hours'] = (df_enhanced['study_hours_per_week'] < 10).astype(int)
    df_enhanced['low_library_usage'] = (df_enhanced['library_visits_per_month'] < 5).astype(int)
    df_enhanced['poor_ct_performance'] = (df_enhanced['ct_average_manual'] < 5).astype(int)
    
    # 13. Excellence Indicators
    df_enhanced['high_performer'] = (df_enhanced['weighted_academic_score'] > 80).astype(int)
    df_enhanced['consistent_performer'] = (df_enhanced['academic_cv'] < 0.2).astype(int)
    df_enhanced['improving_student'] = (df_enhanced['ct_improvement'] > 1).astype(int)
    
    print(f"Enhanced features created: {df_enhanced.shape[1]} total features")
    return df_enhanced

# Apply advanced feature engineering
df_processed = advanced_feature_engineering(df)

# Remove rows with missing target
df_processed = df_processed.dropna(subset=['final_grade'])
print(f"Final dataset shape: {df_processed.shape}")

# Prepare features and target
X = df_processed.drop('final_grade', axis=1)
y = df_processed['final_grade']

print(f"Features: {X.shape[1]} features")
print(f"Target classes: {sorted(y.unique())}")

# Advanced preprocessing
label_encoders = {}
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Categorical features to encode: {len(categorical_features)}")

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Advanced feature selection
print(f"\n=== ADVANCED FEATURE SELECTION ===")

# Method 1: Statistical selection
selector_stats = SelectKBest(score_func=f_classif, k='all')
X_stats = selector_stats.fit_transform(X, y)
feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector_stats.scores_
}).sort_values('score', ascending=False)

print("Top 20 features by statistical score:")
print(feature_scores.head(20))

# Method 2: Tree-based feature importance
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X, y)
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features by Random Forest importance:")
print(feature_importance_rf.head(20))

# Combine both methods for feature selection
top_stats_features = set(feature_scores.head(30)['feature'].tolist())
top_rf_features = set(feature_importance_rf.head(30)['feature'].tolist())
selected_features = list(top_stats_features.union(top_rf_features))

print(f"\nSelected {len(selected_features)} features for modeling")

X_selected = X[selected_features]

# Stratified split with proper class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Advanced scaling
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))
print(f"Class weights: {class_weight_dict}")

# === ADVANCED MODEL ENSEMBLE ===
print(f"\n=== BUILDING ADVANCED MODEL ENSEMBLE ===")

# Model 1: Optimized Random Forest
print("Training Random Forest...")
rf_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.8],
    'class_weight': ['balanced'],
    'bootstrap': [True],
    'oob_score': [True]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
print(f"Best RF parameters: {rf_grid.best_params_}")
print(f"Best RF CV score: {rf_grid.best_score_:.4f}")

# Model 2: Optimized Gradient Boosting
print("Training Gradient Boosting...")
gb_params = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
gb_grid.fit(X_train, y_train)

best_gb = gb_grid.best_estimator_
print(f"Best GB parameters: {gb_grid.best_params_}")
print(f"Best GB CV score: {gb_grid.best_score_:.4f}")

# Model 3: Extra Trees (Extremely Randomized Trees)
print("Training Extra Trees...")
et_params = {
    'n_estimators': [300, 500],
    'max_depth': [15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

et = ExtraTreesClassifier(random_state=42, n_jobs=-1)
et_grid = GridSearchCV(et, et_params, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
et_grid.fit(X_train, y_train)

best_et = et_grid.best_estimator_
print(f"Best ET parameters: {et_grid.best_params_}")
print(f"Best ET CV score: {et_grid.best_score_:.4f}")

# Model 4: Neural Network
print("Training Neural Network...")
mlp_params = {
    'hidden_layer_sizes': [(100, 50), (150, 75), (200, 100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.001, 0.01, 0.1],
    'learning_rate': ['adaptive'],
    'max_iter': [500]
}

mlp = MLPClassifier(random_state=42)
mlp_grid = GridSearchCV(mlp, mlp_params, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
mlp_grid.fit(X_train_scaled, y_train)  # Neural networks need scaled data

best_mlp = mlp_grid.best_estimator_
print(f"Best MLP parameters: {mlp_grid.best_params_}")
print(f"Best MLP CV score: {mlp_grid.best_score_:.4f}")

# Create Voting Ensemble
print("Creating Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb),
        ('et', best_et)
    ],
    voting='soft'  # Use probability voting
)

voting_clf.fit(X_train, y_train)

# Evaluate all models
models = {
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb,
    'Extra Trees': best_et,
    'Neural Network': best_mlp,
    'Voting Ensemble': voting_clf
}

results = {}

print(f"\n=== MODEL EVALUATION RESULTS ===")
for name, model in models.items():
    if name == 'Neural Network':
        # Neural network needs scaled data
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")

# Select best model
best_model_name = max(results.keys(), key=lambda x: results[x]['balanced_accuracy'])
best_model = models[best_model_name]
best_results = results[best_model_name]

print(f"\n=== BEST MODEL: {best_model_name} ===")
print(f"Balanced Accuracy: {best_results['balanced_accuracy']:.4f}")
print(f"Standard Accuracy: {best_results['accuracy']:.4f}")
print(f"F1 Score: {best_results['f1_score']:.4f}")

# Detailed evaluation of best model
print(f"\nDetailed Classification Report for {best_model_name}:")
print(classification_report(y_test, best_results['predictions']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_results['predictions'])
print(f"\nConfusion Matrix:")
print(cm)

# Per-class performance analysis
print(f"\nPer-class Performance Analysis:")
for i, grade in enumerate(sorted(y.unique())):
    if i < len(cm):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"{grade}: Precision={precision:.3f}, Recall={recall:.3f}, Specificity={specificity:.3f}")

# Cross-validation for reliability
cv_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='balanced_accuracy')
print(f"\n10-Fold Cross-Validation Results:")
print(f"Mean Balanced Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Individual CV scores: {cv_scores}")

# Feature importance analysis
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features in Best Model:")
    print(feature_importance.head(15))

# Save the best model and artifacts
model_artifacts = {
    'model_type': best_model_name,
    'feature_names': X_train.columns.tolist(),
    'selected_features': selected_features,
    'label_encoders': {k: v.classes_.tolist() for k, v in label_encoders.items()},
    'scaler_center': scaler.center_.tolist() if hasattr(scaler, 'center_') else None,
    'scaler_scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
    'class_weights': class_weight_dict,
    'feature_importance': feature_importance.to_dict('records') if hasattr(best_model, 'feature_importances_') else None,
    'accuracy_metrics': {
        'balanced_accuracy': best_results['balanced_accuracy'],
        'standard_accuracy': best_results['accuracy'],
        'f1_score': best_results['f1_score'],
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    },
    'grade_mapping': {grade: i for i, grade in enumerate(sorted(y.unique()))},
    'model_comparison': {name: {'balanced_accuracy': results[name]['balanced_accuracy'], 
                               'accuracy': results[name]['accuracy']} for name in results.keys()}
}

# Save model
with open('best_high_accuracy_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model_artifacts_enhanced.json', 'w') as f:
    json.dump(model_artifacts, f, indent=2)

print(f"\n=== HIGH-ACCURACY MODEL TRAINING COMPLETED ===")
print(f"Best Model: {best_model_name}")
print(f"Balanced Accuracy: {best_results['balanced_accuracy']:.4f}")
print(f"Standard Accuracy: {best_results['accuracy']:.4f}")
print(f"Cross-Validation Mean: {cv_scores.mean():.4f}")
print(f"Model saved as 'best_high_accuracy_model.pkl'")
print(f"Artifacts saved as 'model_artifacts_enhanced.json'")

# Performance improvement summary
print(f"\n=== ACCURACY IMPROVEMENTS IMPLEMENTED ===")
print("1. Advanced Feature Engineering: 40+ engineered features")
print("2. Multiple Algorithm Comparison: RF, GB, ET, MLP, Voting Ensemble")
print("3. Hyperparameter Optimization: Grid search for all models")
print("4. Feature Selection: Statistical + Tree-based selection")
print("5. Class Balancing: Balanced class weights")
print("6. Robust Scaling: RobustScaler for outlier handling")
print("7. Cross-Validation: 10-fold CV for reliability")
print("8. Ensemble Methods: Voting classifier for improved accuracy")
print("9. Advanced Metrics: Balanced accuracy, F1-score")
print("10. Interaction Features: Feature combinations for better prediction")
