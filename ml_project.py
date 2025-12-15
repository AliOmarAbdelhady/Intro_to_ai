import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory for all generated files
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_explore_data():
    print("=" * 60)
    print("Loading Mental Productivity Dataset...")
    print("=" * 60)
    
    df = pd.read_csv('mental_productivity_dataset.csv')
    
    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst few rows:")
    print(df.head(10))
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nData types and missing values:")
    print(df.info())
    
    print("\nProductivity score distribution:")
    print(df['productivity_score_1_10'].describe())
    
    return df


def visualize_data(df):
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Target distribution
    plt.subplot(3, 4, 1)
    plt.hist(df['productivity_score_1_10'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Productivity Score Distribution')
    plt.xlabel('Productivity Score (1-10)')
    plt.ylabel('Count')
    
    # Feature distributions
    features = ['sleep_hours', 'daily_exercise_mins', 'screen_time_hours', 'stress_level_1_10', 'mood_level_1_10']
    for idx, feature in enumerate(features, 2):
        plt.subplot(3, 4, idx)
        plt.hist(df[feature], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title(f'{feature.replace("_", " ").title()} Distribution')
        plt.xlabel(feature.replace('_', ' ').title())
        plt.ylabel('Frequency')
    
    # Correlation with target
    plt.subplot(3, 4, 7)
    correlation = df.corr()
    sns.heatmap(correlation[['productivity_score_1_10']].sort_values(by='productivity_score_1_10', ascending=False).head(7),
                annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation with Productivity')
    
    # Box plots
    key_features = ['sleep_hours', 'stress_level_1_10', 'mood_level_1_10']
    df_temp = df.copy()
    df_temp['productivity_binary'] = (df_temp['productivity_score_1_10'] >= df_temp['productivity_score_1_10'].median()).astype(int)
    for idx, feature in enumerate(key_features, 8):
        plt.subplot(3, 4, idx)
        df_temp.boxplot(column=feature, by='productivity_binary', ax=plt.gca())
        plt.title(f'{feature.replace("_", " ").title()} by Productivity')
        plt.suptitle('')
        plt.xlabel('Productivity Level')
        plt.ylabel(feature.replace('_', ' ').title())
    
    # Scatter plot
    plt.subplot(3, 4, 11)
    for prod in [0, 1]:
        prod_data = df_temp[df_temp['productivity_binary'] == prod]
        plt.scatter(prod_data['sleep_hours'], prod_data['mood_level_1_10'], 
                   label=f'Productivity: {"High" if prod else "Low"}', alpha=0.5)
    plt.xlabel('Sleep Hours')
    plt.ylabel('Mood Level')
    plt.title('Sleep vs Mood')
    plt.legend()
    
    # Missing values check
    plt.subplot(3, 4, 12)
    missing = df.isnull().sum()
    plt.bar(range(len(missing)), missing.values)
    plt.title('Missing Values')
    plt.xlabel('Feature Index')
    plt.ylabel('Count')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'data_exploration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {output_path}")


def preprocess_data(df):
    """Handle missing values, outliers, and feature engineering"""
    print("\n" + "=" * 60)
    print("Preprocessing Data...")
    print("=" * 60)
    
    df_processed = df.copy()
    
    # Drop id column
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)
        print("\nDropped id column")
    
    # Check missing values
    print("\nChecking for missing values...")
    missing_values = df_processed.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        print("Missing values handled with median imputation")
    else:
        print("No missing values")
    
    # Remove outliers using IQR
    print("\nHandling outliers...")
    numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('productivity_score_1_10')
    
    outlier_counts = {}
    for feature in numerical_features:
        Q1 = df_processed[feature].quantile(0.25)
        Q3 = df_processed[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_processed[(df_processed[feature] < lower_bound) | 
                               (df_processed[feature] > upper_bound)]
        outlier_counts[feature] = len(outliers)
        
        df_processed[feature] = df_processed[feature].clip(lower_bound, upper_bound)
    
    print(f"Capped outliers in {len([v for v in outlier_counts.values() if v > 0])} features")
    
    # Create new features
    print("\nCreating new features...")
    df_processed['sleep_exercise'] = df_processed['sleep_hours'] * df_processed['daily_exercise_mins']
    df_processed['stress_screen_ratio'] = df_processed['stress_level_1_10'] / (df_processed['screen_time_hours'] + 0.001)
    df_processed['diet_mood_product'] = df_processed['diet_quality_1_10'] * df_processed['mood_level_1_10']
    df_processed['sleep_squared'] = df_processed['sleep_hours'] ** 2
    df_processed['health_score'] = df_processed['sleep_hours'] + df_processed['diet_quality_1_10'] - df_processed['stress_level_1_10']
    
    # Convert to binary classification
    print("\nConverting target to binary (0 = low productivity, 1 = high productivity)...")
    median_productivity = df_processed['productivity_score_1_10'].median()
    df_processed['productivity_binary'] = (df_processed['productivity_score_1_10'] >= median_productivity).astype(int)
    print(f"Median productivity score: {median_productivity:.2f}")
    print(f"Class distribution:\n{df_processed['productivity_binary'].value_counts()}")
    
    X = df_processed.drop(['productivity_score_1_10', 'productivity_binary'], axis=1)
    y = df_processed['productivity_binary']
    
    print(f"\nPreprocessing done. Final shape: {X.shape}")
    
    # Return raw X and y (we will scale later in main)
    return X, y


def build_and_train_models(X_train, X_test, y_train, y_test):
    """Train three different models with overfitting prevention and cross-validation"""
    print("\n" + "=" * 60)
    print("Training Models (with Cross-Validation)...")
    print("=" * 60)
    
    models = {}
    cv_results = {}
    
    # Setup cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Naive Bayes - inherently resistant to overfitting
    print("\nTraining Naive Bayes...")
    nb_model = GaussianNB(var_smoothing=1e-8)  # Slight regularization
    
    # Perform cross-validation
    print("  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(nb_model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results['Naive Bayes'] = cv_scores
    print(f"  CV Scores: {cv_scores}")
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    nb_model.fit(X_train, y_train)
    models['Naive Bayes'] = nb_model
    print("Done")
    
    # Decision Tree - MORE regularization to prevent overfitting
    print("\nTraining Decision Tree (with stronger regularization)...")
    dt_model = DecisionTreeClassifier(
        max_depth=5,              # Reduced from 10 to prevent deep trees
        min_samples_split=50,     # Increased from 20 (need more samples to split)
        min_samples_leaf=25,      # Increased from 10 (larger leaf nodes)
        min_impurity_decrease=0.001,  # Require minimum improvement to split
        max_features='sqrt',      # Use subset of features at each split
        random_state=42,
        ccp_alpha=0.01           # Cost complexity pruning
    )
    
    # Perform cross-validation
    print("  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results['Decision Tree'] = cv_scores
    print(f"  CV Scores: {cv_scores}")
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    print("Done")
    
    # Neural Network - Optimized for best performance
    print("\nTraining Neural Network (optimized for best performance)...")
    ann_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # Larger, deeper network
        activation='relu',
        solver='adam',
        alpha=0.0001,            # Minimal regularization for better fit
        max_iter=2000,           # More training iterations
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,  # Smaller validation = more training data
        n_iter_no_change=50,     # More patience before stopping
        learning_rate_init=0.003, # Higher learning rate
        learning_rate='adaptive',
        batch_size=64,
        tol=1e-6                 # Lower tolerance for convergence
    )
    
    # Perform cross-validation
    print("  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(ann_model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results['ANN'] = cv_scores
    print(f"  CV Scores: {cv_scores}")
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    ann_model.fit(X_train, y_train)
    models['ANN'] = ann_model
    print("Done")
    
    return models, cv_results


def evaluate_models(models, X_train, X_test, y_train, y_test, cv_results):
    """Evaluate model performance"""
    print("\n" + "=" * 60)
    print("Evaluating Models...")
    print("=" * 60)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        cv_mean = cv_results[model_name].mean()
        cv_std = cv_results[model_name].std()
        
        print(f"  Cross-Val Mean:    {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        print(f"  Training accuracy: {train_accuracy:.4f}")
        print(f"  Testing accuracy:  {test_accuracy:.4f}")
        print(f"  Overfit (Train-Test): {(train_accuracy - test_accuracy):.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['Low Productivity', 'High Productivity']))
        
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"Confusion Matrix:")
        print(f"              Predicted")
        print(f"              No   Yes")
        print(f"Actual No   [ {cm[0][0]:3d}   {cm[0][1]:3d} ]")
        print(f"Actual Yes  [ {cm[1][0]:3d}   {cm[1][1]:3d} ]")
        
        results[model_name] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'predictions': y_test_pred,
            'confusion_matrix': cm
        }
    
    return results


def visualize_results(results, y_test):
    """Visualize model comparison"""
    print("\nCreating comparison plots...")
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Accuracy Comparison
    plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    train_accs = [results[m]['train_accuracy'] for m in model_names]
    test_accs = [results[m]['test_accuracy'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, train_accs, width, label='Training', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, test_accs, width, label='Testing', color='#e74c3c', alpha=0.8)
    
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names, rotation=15, ha='right')
    plt.ylim([0.6, 1.0])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(train_accs):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(test_accs):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Test Accuracy Only (clearer comparison)
    plt.subplot(2, 3, 2)
    colors = ['#2ecc71', '#9b59b6', '#f39c12']
    bars = plt.bar(model_names, test_accs, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Test Accuracy', fontweight='bold')
    plt.title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.ylim([0.6, 1.0])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}\n({height*100:.2f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3-5. Confusion Matrices
    for idx, (model_name, result) in enumerate(results.items(), 3):
        plt.subplot(2, 3, idx)
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Low Prod', 'High Prod'],
                   yticklabels=['Low Prod', 'High Prod'])
        plt.title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        plt.ylabel('Actual', fontweight='bold')
        plt.xlabel('Predicted', fontweight='bold')
    
    # 6. Performance Summary Table
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    table_data = []
    for model_name in model_names:
        train_acc = results[model_name]['train_accuracy']
        test_acc = results[model_name]['test_accuracy']
        overfit = train_acc - test_acc
        table_data.append([model_name, f'{test_acc:.4f}', f'{train_acc:.4f}', f'{overfit:.4f}'])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Model', 'Test Acc', 'Train Acc', 'Overfit'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best model
    best_idx = test_accs.index(max(test_accs))
    for i in range(4):
        table[(best_idx + 1, i)].set_facecolor('#2ecc71')
        table[(best_idx + 1, i)].set_alpha(0.3)
    
    plt.title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {output_path}")


def save_best_model(models, results, scaler, X):
    """Save the best performing model and scaler"""
    print("\n" + "=" * 60)
    print("Saving Best Model...")
    print("=" * 60)
    
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    best_model = models[best_model_name]
    best_acc = results[best_model_name]['test_accuracy']
    
    # Save model
    model_filename = OUTPUT_DIR / 'best_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"\nSaved {best_model_name} to: {model_filename}")
    print(f"Test accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    # Save scaler
    scaler_filename = OUTPUT_DIR / 'scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"Saved scaler to: {scaler_filename}")
    
    # Save feature columns (needed by app.py)
    feature_columns = X.columns.tolist() if hasattr(X, 'columns') else []
    features_filename = OUTPUT_DIR / 'feature_columns.csv'
    pd.DataFrame(feature_columns, columns=['feature']).to_csv(features_filename, index=False)
    print(f"Saved feature columns to: {features_filename}")
    
    # Save model info
    info_filename = OUTPUT_DIR / 'model_info.txt'
    with open(info_filename, 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
        f.write(f"Training Accuracy: {results[best_model_name]['train_accuracy']:.4f}\n")
        f.write(f"\nModel saved at: {model_filename}\n")
        f.write(f"Scaler saved at: {scaler_filename}\n")
        f.write(f"Features saved at: {features_filename}\n")
    print(f"Saved model info to: {info_filename}")
    
    print("\n" + "=" * 60)


def print_final_results(results):
    """Print summary of results"""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    best_name = best_model[0]
    best_acc = best_model[1]['test_accuracy']
    best_cv = best_model[1]['cv_mean']
    
    print(f"\nBest performing model: {best_name}")
    print(f"  Cross-validation: {best_cv:.4f} (+/- {best_model[1]['cv_std'] * 2:.4f})")
    print(f"  Test accuracy:    {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    print("\nAll models ranked by test accuracy:")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    for rank, (name, result) in enumerate(sorted_models, 1):
        cv_mean = result['cv_mean']
        test_acc = result['test_accuracy']
        train_acc = result['train_accuracy']
        overfit = train_acc - test_acc
        print(f"   {rank}. {name:20s} - CV: {cv_mean:.4f} | Test: {test_acc:.4f} | Train: {train_acc:.4f} | Overfit: {overfit:+.4f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("Mental Productivity Prediction - ML Project")
    print("=" * 60)
    
    # Load data
    df = load_and_explore_data()
    visualize_data(df)
    
   # Preprocess (Get raw data)
    X, y = preprocess_data(df)
    
    # Split data FIRST
    print("\nSplitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Scale SECOND (Fit on Train, Transform Test)
    print("Scaling features...")
    scaler = StandardScaler()
    
    # The fix: fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep column names
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    
    # Train models with cross-validation
    models, cv_results = build_and_train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate
    results = evaluate_models(models, X_train, X_test, y_train, y_test, cv_results)
    
    # Visualize results
    visualize_results(results, y_test)
    
    # Save best model
    save_best_model(models, results, scaler, X)
    
    print_final_results(results)
    
    print(f"\nOutput files saved in: {OUTPUT_DIR.absolute()}")
    print("  - data_exploration.png")
    print("  - model_comparison.png")
    print("  - best_model.pkl")
    print("  - scaler.pkl")
    print("  - feature_columns.csv")
    print("  - model_info.txt")


if __name__ == "__main__":
    main()
