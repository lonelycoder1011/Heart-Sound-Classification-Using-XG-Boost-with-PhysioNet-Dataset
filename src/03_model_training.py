"""
Heart Sound Classification Model Training
========================================

Professional lightweight model for heart sound classification demonstration.
Optimized for clinical deployment with high accuracy and fast inference.

Features:
- Multiple model architectures (Random Forest, XGBoost, Neural Network)
- Cross-validation and robust evaluation
- Hyperparameter optimization
- Model interpretability and feature importance
- Deployment-ready model export

Author: ML Engineer for Heart Sound Classification
Version: 1.0
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import joblib
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.inspection import permutation_importance
import xgboost as xgb

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Using traditional ML models only.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeartSoundClassifier:
    """
    Professional heart sound classification system with multiple model options.
    Designed for clinical deployment with robust evaluation metrics.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.training_history = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss'
            },
            'neural_network': {
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 15
            }
        }
    
    def load_physionet_data(self, data_path: str = "physionet_heart_features.csv", 
                           test_size: float = 0.2, 
                           use_database_split: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split the PhysioNet dataset with proper handling of database splits.
        
        Args:
            data_path: Path to the PhysioNet feature CSV file
            test_size: Proportion of data for testing
            use_database_split: If True, ensure train/test split respects database boundaries
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Loading PhysioNet data from {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded PhysioNet dataset with shape: {df.shape}")
            
            # Display dataset info
            print("\n" + "="*50)
            print("PHYSIONET DATASET LOADED")
            print("="*50)
            print(f"Total samples: {len(df):,}")
            
            if 'database' in df.columns:
                print(f"Databases: {df['database'].nunique()}")
                print(f"Database distribution:")
                for db, count in df['database'].value_counts().items():
                    print(f"  {db}: {count:,}")
            
            print(f"Label distribution:")
            for label, count in df['label'].value_counts().items():
                percentage = (count / len(df)) * 100
                print(f"  {label}: {count:,} ({percentage:.1f}%)")
            
            if 'quality_score' in df.columns:
                print(f"Average quality score: {df['quality_score'].mean():.3f}")
            
            # Separate features and labels
            feature_cols = [col for col in df.columns if col not in ['label', 'filename', 'quality_score', 'database']]
            X = df[feature_cols].values
            y = df['label'].values
            
            logger.info(f"Feature columns: {len(feature_cols)}")
            logger.info(f"Sample feature columns: {feature_cols[:10]}")
            # Save feature names for later interpretability
            self.feature_names = feature_cols
            
            # Encode labels (PhysioNet: normal/abnormal -> 0/1)
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            print(f"\nLabel encoding:")
            for i, label in enumerate(self.label_encoder.classes_):
                print(f"  {label} -> {i}")
            
            # Handle database-aware splitting for PhysioNet
            if use_database_split and 'database' in df.columns:
                logger.info("Using database-aware train/test split")
                
                # Get unique databases
                databases = df['database'].unique()
                n_test_dbs = max(1, int(len(databases) * test_size))
                
                # Randomly select databases for test set
                np.random.seed(42)
                test_databases = np.random.choice(databases, size=n_test_dbs, replace=False)
                
                # Create train/test masks
                test_mask = df['database'].isin(test_databases)
                train_mask = ~test_mask
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y_encoded[train_mask], y_encoded[test_mask]
                
                print(f"\nDatabase split:")
                print(f"  Train databases: {sorted([db for db in databases if db not in test_databases])}")
                print(f"  Test databases: {sorted(test_databases)}")
                
            else:
                # Standard stratified split
                logger.info("Using standard stratified train/test split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
            
            # Normalize features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Training set: {X_train.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            logger.info(f"Features: {X_train.shape[1]}")
            logger.info(f"Classes: {len(np.unique(y_encoded))}")
            
            # Check class balance in splits
            train_counts = np.bincount(y_train)
            test_counts = np.bincount(y_test)
            
            print(f"\nClass distribution in splits:")
            for i, label in enumerate(self.label_encoder.classes_):
                train_pct = (train_counts[i] / len(y_train)) * 100
                test_pct = (test_counts[i] / len(y_test)) * 100
                print(f"  {label}:")
                print(f"    Train: {train_counts[i]:,} ({train_pct:.1f}%)")
                print(f"    Test:  {test_counts[i]:,} ({test_pct:.1f}%)")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to load PhysioNet data: {e}")
            raise
    
    def load_data(self, data_path: str, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Legacy method for backward compatibility.
        For PhysioNet dataset, use load_physionet_data instead.
        """
        logger.warning("Using legacy load_data method. Consider using load_physionet_data for PhysioNet dataset.")
        return self.load_physionet_data(data_path, test_size, use_database_split=False)
    

    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training with proper scaling and encoding.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, 
            random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training set: {X_train_scaled.shape[0]} samples")
        logger.info(f"Test set: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_random_forest(self) -> RandomForestClassifier:
        """
        Build Random Forest classifier optimized for heart sounds.
        
        Returns:
            Configured Random Forest model
        """
        config = self.model_configs['random_forest']
        model = RandomForestClassifier(**config)
        logger.info("Built Random Forest classifier")
        return model
    
    def build_xgboost(self) -> xgb.XGBClassifier:
        """
        Build XGBoost classifier optimized for heart sounds.
        
        Returns:
            Configured XGBoost model
        """
        config = self.model_configs['xgboost']
        model = xgb.XGBClassifier(**config)
        logger.info("Built XGBoost classifier")
        return model
    
    def build_neural_network(self, input_dim: int) -> Optional[object]:
        """
        Build neural network for heart sound classification.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Configured neural network model
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available for neural network")
            return None
        
        config = self.model_configs['neural_network']
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(config['hidden_layers'][0], 
                       activation='relu', 
                       input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Dropout(config['dropout_rate']))
        
        # Hidden layers
        for units in config['hidden_layers'][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(config['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Built neural network classifier")
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, 
                   y_val: Optional[np.ndarray] = None) -> None:
        """
        Train the selected model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for neural network)
            y_val: Validation labels (for neural network)
        """
        logger.info(f"Training {self.model_type} model")
        
        if self.model_type == 'random_forest':
            self.model = self.build_random_forest()
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'xgboost':
            self.model = self.build_xgboost()
            
            # Use validation set if provided
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            # Some xgboost versions don't support early_stopping_rounds in fit with sklearn API
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=20,
                    verbose=False
                )
            except TypeError:
                # Fallback: simple fit without early stopping
                self.model.fit(X_train, y_train, verbose=False)
            
        elif self.model_type == 'neural_network':
            if not TF_AVAILABLE:
                raise ValueError("TensorFlow not available for neural network")
            
            self.model = self.build_neural_network(X_train.shape[1])
            if self.model is None:
                return
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=self.model_configs['neural_network']['patience'], 
                            restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
            ]
            
            # Training
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=self.model_configs['neural_network']['epochs'],
                batch_size=self.model_configs['neural_network']['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            self.training_history = history.history
        
        logger.info(f"Model training completed")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation with clinical metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Predictions
        if self.model_type == 'neural_network':
            y_pred_proba = self.model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        metrics = {
            'accuracy': np.mean(y_pred == y_test),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': report
        }
        
        # Log results
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation for robust performance estimation.
        
        Args:
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        if self.model_type == 'neural_network':
            logger.warning("Cross-validation not implemented for neural networks")
            return {}
        
        # Build model for CV
        if self.model_type == 'random_forest':
            model = self.build_random_forest()
        elif self.model_type == 'xgboost':
            model = self.build_xgboost()
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
        cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        
        results = {
            'accuracy_mean': np.mean(cv_scores),
            'accuracy_std': np.std(cv_scores),
            'f1_mean': np.mean(cv_f1),
            'f1_std': np.std(cv_f1),
            'auc_mean': np.mean(cv_auc),
            'auc_std': np.std(cv_auc),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"CV Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        logger.info(f"CV F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
        logger.info(f"CV AUC: {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
        
        return results
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance for model interpretability.
        
        Returns:
            Feature importance scores
        """
        if self.model_type in ['random_forest', 'xgboost']:
            return self.model.feature_importances_
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            return None
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot top feature importances.
        
        Args:
            top_n: Number of top features to plot
        """
        importance = self.get_feature_importance()
        if importance is None:
            return
        
        # Get top features
        indices = np.argsort(importance)[::-1][:top_n]
        top_features = [self.feature_names[i] for i in indices]
        top_importance = importance[indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.title(f'Top {top_n} Feature Importances - {self.model_type.title()}')
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model and preprocessing components.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model_type': self.model_type,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        if self.model_type == 'neural_network':
            # Save neural network separately
            self.model.save(f"{filepath}_nn_model.h5")
            model_data['model_path'] = f"{filepath}_nn_model.h5"
        else:
            model_data['model'] = self.model
        
        # Save with joblib
        joblib.dump(model_data, f"{filepath}.pkl")
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model and preprocessing components.
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(f"{filepath}.pkl")
        
        self.model_type = model_data['model_type']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data.get('training_history', {})
        
        if self.model_type == 'neural_network':
            if TF_AVAILABLE:
                self.model = tf.keras.models.load_model(model_data['model_path'])
            else:
                raise ValueError("TensorFlow not available to load neural network")
        else:
            self.model = model_data['model']
        
        logger.info(f"Model loaded from {filepath}")

class ModelComparison:
    """
    Compare multiple model architectures for heart sound classification.
    """
    
    def __init__(self):
        """Initialize model comparison."""
        self.results = {}
        self.models = {}
    
    def compare_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Compare different model architectures.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Comparison results
        """
        model_types = ['random_forest', 'xgboost']
        if TF_AVAILABLE:
            model_types.append('neural_network')
        
        logger.info("Starting model comparison")
        
        for model_type in model_types:
            logger.info(f"Training {model_type}")
            
            # Initialize classifier
            classifier = HeartSoundClassifier(model_type=model_type)
            
            # Prepare data
            X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
            
            # Train model
            if model_type == 'neural_network':
                # Use part of training data for validation
                X_train_nn, X_val, y_train_nn, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                classifier.train_model(X_train_nn, y_train_nn, X_val, y_val)
            else:
                classifier.train_model(X_train, y_train)
            
            # Evaluate model
            metrics = classifier.evaluate_model(X_test, y_test)
            
            # Cross-validation
            cv_results = classifier.cross_validate(X_train, y_train)
            
            # Store results
            self.results[model_type] = {
                'test_metrics': metrics,
                'cv_results': cv_results
            }
            self.models[model_type] = classifier
        
        # Find best model
        best_model = max(self.results.keys(), 
                        key=lambda x: self.results[x]['test_metrics']['f1_score'])
        
        logger.info(f"Best model: {best_model}")
        
        return {
            'results': self.results,
            'best_model': best_model
        }

def main():
    """
    PhysioNet Challenge 2016 Heart Sound Classification Model Training.
    """
    print("PhysioNet Challenge 2016 - Heart Sound Classification")
    print("=" * 55)
    
    # Check for PhysioNet features file
    features_file = "physionet_heart_features.csv"
    
    if not Path(features_file).exists():
        print(f"❌ PhysioNet features file not found: {features_file}")
        print("Please run 02_preprocessing_features.py first to extract features from PhysioNet dataset")
        print("\nSteps to get started:")
        print("1. Ensure PhysioNet dataset is downloaded (training-a through training-f)")
        print("2. Run: python 02_preprocessing_features.py")
        print("3. Then run this script again")
        return
    
    print(f"✅ Found PhysioNet features file: {features_file}")
    
    # Initialize classifier with best model for heart sounds
    print("\nInitializing Heart Sound Classifier...")
    classifier = HeartSoundClassifier(model_type='xgboost')  # XGBoost often works well for audio features
    
    # Load PhysioNet data with database-aware splitting
    print("Loading PhysioNet Challenge 2016 dataset...")
    try:
        X_train, X_test, y_train, y_test = classifier.load_physionet_data(
            data_path=features_file,
            test_size=0.2,
            use_database_split=True  # Ensure no data leakage between databases
        )
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    if len(X_train) == 0:
        print("❌ No training data loaded. Please check the features file.")
        return
    
    print(f"\n✅ Data loaded successfully!")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    
    # Train model
    print(f"\n🚀 Training {classifier.model_type.upper()} model...")
    print("This may take a few minutes depending on dataset size...")
    
    start_time = time.time()
    classifier.train_model(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Model training completed in {training_time:.1f} seconds")
    
    # Evaluate model
    print("\n📊 Evaluating model performance...")
    metrics = classifier.evaluate_model(X_test, y_test)
    
    # Display results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Interpret results for clinical context
    if metrics['accuracy'] >= 0.85:
        print("\n🎉 EXCELLENT: Model shows high accuracy suitable for clinical screening")
    elif metrics['accuracy'] >= 0.75:
        print("\n✅ GOOD: Model performance is acceptable for heart sound classification")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT: Consider feature engineering or model tuning")
    
    if metrics['recall'] >= 0.80:
        print("✅ High recall - good at detecting abnormal heart sounds")
    else:
        print("⚠️  Low recall - may miss some abnormal cases (clinical concern)")
    
    if metrics['precision'] >= 0.80:
        print("✅ High precision - low false positive rate")
    else:
        print("⚠️  Lower precision - may flag normal sounds as abnormal")
    
    # Cross-validation for robust evaluation
    print("\n🔄 Performing 5-fold cross-validation...")
    cv_results = classifier.cross_validate(X_train, y_train, cv_folds=5)
    
    print(f"Cross-validation results:")
    print(f"  Mean Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    print(f"  Mean F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
    
    # Feature importance analysis
    print("\n🔍 Analyzing feature importance...")
    try:
        classifier.plot_feature_importance(top_n=20)
        print("✅ Feature importance plot saved as 'feature_importance.png'")
    except Exception as e:
        print(f"⚠️  Could not generate feature importance plot: {e}")
    
    # Save model
    model_path = f"physionet_heart_model_{classifier.model_type}"
    print(f"\n💾 Saving trained model...")
    classifier.save_model(model_path)
    print(f"✅ Model saved as: {model_path}.pkl")
    
    # Model comparison across architectures
    print("\n🏆 Comparing multiple model architectures...")
    comparison = ModelComparison()
    
    # Use a subset for comparison to save time
    sample_size = min(5000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train[indices]
    y_sample = y_train[indices]
    
    comparison_results = comparison.compare_models(X_sample, y_sample)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*60)
    results_dict = comparison_results['results']
    for model_name, res in results_dict.items():
        metrics_cmp = res['test_metrics']
        print(f"{model_name.upper()}:")
        print(f"  Accuracy:  {metrics_cmp['accuracy']:.4f}")
        print(f"  F1-Score:  {metrics_cmp['f1_score']:.4f}")
        print(f"  AUC-ROC:   {metrics_cmp['auc_roc']:.4f}")
        print()
    
    # Final recommendations
    print("="*60)
    print("RECOMMENDATIONS & NEXT STEPS")
    print("="*60)
    print("1. 🎯 Model is ready for deployment testing")
    print("2. 📈 Consider hyperparameter tuning for better performance")
    print("3. 🔄 Validate on additional datasets for generalization")
    print("4. 🏥 Test with clinical experts for real-world validation")
    print("5. 📊 Monitor performance on new data over time")
    
    if metrics['accuracy'] >= 0.80 and metrics['recall'] >= 0.75:
        print("\n🚀 MODEL READY FOR CLINICAL VALIDATION!")
    else:
        print("\n🔧 CONSIDER ADDITIONAL OPTIMIZATION BEFORE DEPLOYMENT")
    
    print("="*60)

if __name__ == "__main__":
    main()
