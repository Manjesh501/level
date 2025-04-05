import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import joblib
import json
from app.ml.classification.preprocess import clean_text
from app.ml.classification.augment import augment_minority_classes

def train_model():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load data with absolute paths
    print("Loading data...")
    data_path = os.path.join(current_dir, "../../../data/naac_documents.csv")
    labels_path = os.path.join(current_dir, "../../../data/labels.json")
    
    # Load and clean data
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text', 'label_criterion', 'label_metric'])
    
    with open(labels_path, "r") as f:
        labels_map = json.load(f)
    
    # Convert numeric columns to string
    df['label_criterion'] = df['label_criterion'].astype(str)
    df['label_metric'] = df['label_metric'].astype(str)
    
    # Create labels
    df['label'] = df['label_criterion'] + ' - ' + df['label_metric']
    df['criterion_label'] = 'Criterion ' + df['label_criterion']
    
    # Print distributions
    print("\nInitial class distribution:")
    print(df['criterion_label'].value_counts())
    
    # Preprocess text
    print("\nPreprocessing text...")
    df['text'] = df['text'].apply(clean_text)
    
    # Augment data
    print("\nAugmenting minority classes...")
    df = augment_minority_classes(df, min_samples=10)
    
    print("\nFinal class distribution:")
    print(df['criterion_label'].value_counts())
    
    # Split data
    X = df['text'].values
    y = df['criterion_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create ensemble pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 4),
            min_df=1,
            max_df=0.95,
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )),
        ('classifier', VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            ('svm', LinearSVC(
                C=1.0,
                class_weight='balanced',
                random_state=42,
                max_iter=2000
            )),
            ('xgb', XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ))
        ], voting='hard'))
    ])
    
    # Train
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    accuracy = pipeline.score(X_test, y_test)
    print(f"\nAccuracy: {accuracy:.2f}")
    
    # Classification report
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # Save model
    try:
        models_dir = os.path.join(current_dir, "../../../models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, "document_classifier.pkl")
        model_artifacts = {
            'pipeline': pipeline,
            'labels_map': labels_map
        }
        
        joblib.dump(model_artifacts, model_path)
        print(f"\nModel artifacts saved to: {model_path}")
        
    except Exception as e:
        print(f"\nError saving model: {str(e)}")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    train_model()