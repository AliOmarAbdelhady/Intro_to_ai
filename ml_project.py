import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_explore_data():
    """Load and explore the heart disease dataset"""
    print("=" * 60)
    print("Loading Heart Disease Dataset...")
    print("=" * 60)
    
    df = pd.read_csv('heart.csv')
    
    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst few rows:")
    print(df.head(10))
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nData types and missing values:")
    print(df.info())
    
    print("\nTarget distribution:")
    print(df['target'].value_counts().sort_index())
    
    return df


def visualize_data(df):
    """Create visualizations for data exploration"""
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Target distribution
    plt.subplot(3, 4, 1)
    df['target'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Disease Distribution')
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Feature distributions
    features = ['age', 'trestbps', 'chol', 'thalach']
    for idx, feature in enumerate(features, 2):
        plt.subplot(3, 4, idx)
        plt.hist(df[feature], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title(f'{feature.title()} Distribution')
        plt.xlabel(feature.title())
        plt.ylabel('Frequency')
    
    # Correlation with target
    plt.subplot(3, 4, 6)
    correlation = df.corr()
    sns.heatmap(correlation[['target']].sort_values(by='target', ascending=False).head(6),
                annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation with Target')
    
    # Box plots
    key_features = ['age', 'trestbps', 'chol', 'thalach']
    for idx, feature in enumerate(key_features, 7):
        plt.subplot(3, 4, idx)
        df.boxplot(column=feature, by='target', ax=plt.gca())
        plt.title(f'{feature.title()} by Disease')
        plt.suptitle('')
        plt.xlabel('Disease Status')
        plt.ylabel(feature.title())
    
    # Scatter plot
    plt.subplot(3, 4, 11)
    df_temp = df.copy()
    df_temp['disease_binary'] = (df_temp['target'] > 0).astype(int)
    for disease in [0, 1]:
        disease_data = df_temp[df_temp['disease_binary'] == disease]
        plt.scatter(disease_data['age'], disease_data['thalach'], 
                   label=f'Disease: {"Yes" if disease else "No"}', alpha=0.5)
    plt.xlabel('Age')
    plt.ylabel('Max Heart Rate')
    plt.title('Age vs Heart Rate')
    plt.legend()
    
    # Missing values check
    plt.subplot(3, 4, 12)
    missing = df.isnull().sum()
    plt.bar(range(len(missing)), missing.values)
    plt.title('Missing Values')
    plt.xlabel('Feature Index')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('c:/Users/aliom/Intro_to_ai/data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved data_exploration.png")


def preprocess_data(df):
    """Handle missing values, outliers, and feature engineering"""
    print("\n" + "=" * 60)
    print("Preprocessing Data...")
    print("=" * 60)
    
    df_processed = df.copy()
    
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
    numerical_features.remove('target')
    
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
    df_processed['age_thalach'] = df_processed['age'] * df_processed['thalach']
    df_processed['bp_chol_ratio'] = df_processed['trestbps'] / (df_processed['chol'] + 0.001)
    df_processed['age_squared'] = df_processed['age'] ** 2
    df_processed['exercise_heart_ratio'] = df_processed['exang'] * df_processed['thalach']
    
    # Convert to binary classification
    print("\nConverting target to binary (0 = no disease, 1 = disease)...")
    df_processed['disease_binary'] = (df_processed['target'] > 0).astype(int)
    print(f"Class distribution:\n{df_processed['disease_binary'].value_counts()}")
    
    X = df_processed.drop(['target', 'disease_binary'], axis=1)
    y = df_processed['disease_binary']
    
    # Scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"\nPreprocessing done. Final shape: {X_scaled.shape}")
    
    return X_scaled, y, scaler


def build_and_train_models(X_train, X_test, y_train, y_test):
    """Train three different models"""
    print("\n" + "=" * 60)
    print("Training Models...")
    print("=" * 60)
    
    models = {}
    
    # Naive Bayes
    print("\nTraining Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    models['Naive Bayes'] = nb_model
    print("Done")
    
    # Decision Tree
    print("\nTraining Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=20, 
                                     min_samples_leaf=10, random_state=42)
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    print("Done")
    
    # Neural Network
    print("\nTraining Neural Network (64-32-16 architecture)...")
    ann_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), 
                             activation='relu',
                             solver='adam',
                             max_iter=500,
                             random_state=42,
                             early_stopping=True,
                             validation_fraction=0.1)
    ann_model.fit(X_train, y_train)
    models['ANN'] = ann_model
    print("Done")
    
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
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
        
        print(f"  Training accuracy: {train_accuracy:.4f}")
        print(f"  Testing accuracy:  {test_accuracy:.4f}")
        print(f"  Difference:        {(train_accuracy - test_accuracy):.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['No Disease', 'Disease']))
        
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"Confusion Matrix:")
        print(f"              Predicted")
        print(f"              No   Yes")
        print(f"Actual No   [ {cm[0][0]:3d}   {cm[0][1]:3d} ]")
        print(f"Actual Yes  [ {cm[1][0]:3d}   {cm[1][1]:3d} ]")
        
        results[model_name] = {
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
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
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
    plt.savefig('c:/Users/aliom/Intro_to_ai/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved model_comparison.png")


def print_final_results(results):
    """Print summary of results"""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    best_name = best_model[0]
    best_acc = best_model[1]['test_accuracy']
    
    print(f"\nBest performing model: {best_name}")
    print(f"Test accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    print("\nAll models ranked:")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    for rank, (name, result) in enumerate(sorted_models, 1):
        test_acc = result['test_accuracy']
        train_acc = result['train_accuracy']
        print(f"   {rank}. {name:20s} - Test: {test_acc:.4f} | Train: {train_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("Heart Disease Prediction - ML Project")
    print("=" * 60)
    
    # Load data
    df = load_and_explore_data()
    visualize_data(df)
    
    # Preprocess
    X, y, scaler = preprocess_data(df)
    
    # Split data
    print("\nSplitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train models
    models = build_and_train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Visualize results
    visualize_results(results, y_test)
    print_final_results(results)
    
    print("\nOutput files:")
    print("  - data_exploration.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()
