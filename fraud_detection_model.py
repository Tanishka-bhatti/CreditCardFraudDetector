"""
Fraud Detection Model Implementation
A comprehensive machine learning pipeline for fraud detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """
    A comprehensive fraud detection model with preprocessing, training, and evaluation capabilities.
    """
    
    def __init__(self):
        """Initialize the fraud detection model."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        self.feature_importance = None
        
    def load_data(self, file_path):
        """
        Load the dataset from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading data from {file_path}...")
            self.data = pd.read_csv(file_path)
            print(f"Dataset loaded successfully! Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Explore the dataset and display basic information."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        print("\n" + "="*50)
        print("DATASET EXPLORATION")
        print("="*50)
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\nData types:")
        print(self.data.dtypes.value_counts())
        
        # Missing values
        print("\nMissing values:")
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")
        
        # Target variable analysis
        target_candidates = ['is_fraud', 'fraud', 'target', 'label', 'class']
        target_col = None
        
        for col in target_candidates:
            if col in self.data.columns:
                target_col = col
                break
        
        if target_col:
            print(f"\nTarget variable '{target_col}' distribution:")
            print(self.data[target_col].value_counts())
            print(f"Fraud rate: {self.data[target_col].mean():.4f}")
        
        # Numeric columns summary
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(f"\nNumeric columns ({len(numeric_cols)}): {list(numeric_cols)}")
        
        # Categorical columns summary
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        print(f"Categorical columns ({len(categorical_cols)}): {list(categorical_cols)}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(self.data.head())
    
    def preprocess_data(self, target_column=None):
        """
        Preprocess the data for machine learning.
        
        Args:
            target_column (str): Name of the target column. If None, auto-detect.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Auto-detect target column if not specified
        if target_column is None:
            target_candidates = ['is_fraud', 'fraud', 'target', 'label', 'class']
            for col in target_candidates:
                if col in self.data.columns:
                    target_column = col
                    break
            
            if target_column is None:
                print("Could not auto-detect target column. Please specify manually.")
                return
        
        print(f"Using target column: {target_column}")
        
        # Separate features and target
        self.target_column = target_column
        y = self.data[target_column]
        X = self.data.drop(columns=[target_column])
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"Encoding {len(categorical_cols)} categorical columns...")
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle datetime columns
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            print(f"Processing {len(datetime_cols)} datetime columns...")
            for col in datetime_cols:
                X[col + '_year'] = X[col].dt.year
                X[col + '_month'] = X[col].dt.month
                X[col + '_day'] = X[col].dt.day
                X[col + '_hour'] = X[col].dt.hour
                X[col + '_dayofweek'] = X[col].dt.dayofweek
                X = X.drop(columns=[col])
        
        # Store processed data
        self.X = X
        self.y = y
        
        print(f"Preprocessed data shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        if self.X is None or self.y is None:
            print("No preprocessed data found. Please preprocess data first.")
            return
            
        print("\n" + "="*50)
        print("DATA SPLITTING")
        print("="*50)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training fraud rate: {self.y_train.mean():.4f}")
        print(f"Test fraud rate: {self.y_test.mean():.4f}")
    
    def handle_imbalance(self, method='smote'):
        """
        Handle class imbalance using various techniques.
        
        Args:
            method (str): Method to use ('smote', 'undersample', 'class_weight', or 'none')
        """
        if self.X_train is None:
            print("No training data found. Please split data first.")
            return
            
        print("\n" + "="*50)
        print(f"HANDLING CLASS IMBALANCE - {method.upper()}")
        print("="*50)
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )
            print(f"After SMOTE - Training set shape: {self.X_train_balanced.shape}")
            print(f"After SMOTE - Fraud rate: {self.y_train_balanced.mean():.4f}")
            
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            self.X_train_balanced, self.y_train_balanced = undersampler.fit_resample(
                self.X_train_scaled, self.y_train
            )
            print(f"After undersampling - Training set shape: {self.X_train_balanced.shape}")
            print(f"After undersampling - Fraud rate: {self.y_train_balanced.mean():.4f}")
            
        elif method == 'class_weight':
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(self.y_train), y=self.y_train
            )
            self.class_weights = dict(zip(np.unique(self.y_train), class_weights))
            self.X_train_balanced = self.X_train_scaled
            self.y_train_balanced = self.y_train
            print(f"Using class weights: {self.class_weights}")
            
        else:  # 'none'
            self.X_train_balanced = self.X_train_scaled
            self.y_train_balanced = self.y_train
            print("No balancing applied.")
    
    def train_models(self):
        """Train multiple machine learning models."""
        if self.X_train_balanced is None:
            print("No balanced training data found. Please handle class imbalance first.")
            return
            
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier( n_estimators=50,        # fewer trees
            max_depth=10,           # shallower trees
            n_jobs=-1,              # use all cores
            random_state=42,
            verbose=1 ),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3,learning_rate=0.1,random_state=42),
            #'SVM': SVC(random_state=42, probability=True)
        }
        
        # Add class weights if using class_weight method
        if hasattr(self, 'class_weights'):
            models['Logistic Regression'].set_params(class_weight=self.class_weights)
            models['Random Forest'].set_params(class_weight=self.class_weights)
            #models['SVM'].set_params(class_weight=self.class_weights)
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                model.fit(self.X_train_balanced, self.y_train_balanced)
                self.models[name] = model
                print(f"{name} trained successfully!")
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        print(f"\nSuccessfully trained {len(self.models)} models!")
    
    def evaluate_models(self):
        """
        Evaluate all trained models and return results.
        
        Returns:
            dict: Dictionary containing evaluation results for each model
        """
        if not self.models:
            print("No trained models found. Please train models first.")
            return {}
            
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score,
                'classification_report': classification_report(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"AUC Score: {auc_score:.4f}")
        
        self.results = results
        
        # Display summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for name, result in results.items():
            print(f"{name}: AUC = {result['auc_score']:.4f}")
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['auc_score'])
        print(f"\nBest model: {best_model} (AUC: {results[best_model]['auc_score']:.4f})")
        
        return results
    
    def plot_results(self):
        """Plot evaluation results including ROC curves and confusion matrices."""
        if not self.results:
            print("No evaluation results found. Please evaluate models first.")
            return
            
        print("\n" + "="*50)
        print("GENERATING PLOTS")
        print("="*50)
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Create subplots
        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        # Plot ROC curves and confusion matrices
        for i, (name, result) in enumerate(self.results.items()):
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            axes[0, i].plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})')
            axes[0, i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, i].set_xlabel('False Positive Rate')
            axes[0, i].set_ylabel('True Positive Rate')
            axes[0, i].set_title(f'ROC Curve - {name}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Confusion Matrix
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, i])
            axes[1, i].set_title(f'Confusion Matrix - {name}')
            axes[1, i].set_xlabel('Predicted')
            axes[1, i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plots saved as 'fraud_detection_results.png'")
    
    def get_feature_importance(self, model_name=None):
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name (str): Name of the model. If None, use the best model.
            
        Returns:
            pandas.DataFrame: Feature importance dataframe
        """
        if not self.results:
            print("No evaluation results found. Please evaluate models first.")
            return None
            
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        
        model = self.results[model_name]['model']
        
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nFeature Importance ({model_name}):")
            print(importance_df.head(10))
            
            self.feature_importance = importance_df
            return importance_df
        else:
            print(f"Model {model_name} does not support feature importance.")
            return None
    
    def save_model(self, filename):
        """
        Save the best model to a file.
        
        Args:
            filename (str): Filename to save the model
        """
        if not self.results:
            print("No evaluation results found. Please evaluate models first.")
            return
            
        # Get the best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        best_model = self.results[best_model_name]['model']
        
        # Save model and scaler
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': list(self.X.columns),
            'target_column': self.target_column,
            'model_name': best_model_name,
            'auc_score': self.results[best_model_name]['auc_score']
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as '{filename}'")
        print(f"Best model: {best_model_name} (AUC: {self.results[best_model_name]['auc_score']:.4f})")
    
    def load_model(self, filename):
        """
        Load a saved model from a file.
        
        Args:
            filename (str): Filename of the saved model
        """
        try:
            model_data = joblib.load(filename)
            self.models[model_data['model_name']] = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.target_column = model_data['target_column']
            print(f"Model loaded from '{filename}'")
            print(f"Model: {model_data['model_name']} (AUC: {model_data['auc_score']:.4f})")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, X_new):
        """
        Make predictions on new data.
        
        Args:
            X_new (pandas.DataFrame): New data to predict
            
        Returns:
            numpy.ndarray: Predictions
        """
        if not self.models:
            print("No trained models found. Please train or load a model first.")
            return None
            
        # Get the best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        model = self.results[best_model_name]['model']
        
        # Preprocess new data
        X_processed = X_new.copy()
        
        # Apply label encoders
        for col, le in self.label_encoders.items():
            if col in X_processed.columns:
                X_processed[col] = le.transform(X_processed[col].astype(str))
        
        # Scale the data
        X_scaled = self.scaler.transform(X_processed)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
