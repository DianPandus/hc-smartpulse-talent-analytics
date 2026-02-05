"""
Model Training Module for HC-SmartPulse
Implements XGBoost classifier with hyperparameter tuning and SHAP explainability
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import shap
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """Train and evaluate XGBoost model for employee attrition prediction"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.best_params = None
        self.shap_explainer = None
        self.shap_values = None
        
    def train_xgboost(self, X_train, y_train, hyperparameter_tuning=True):
        """Train XGBoost classifier with optional hyperparameter tuning"""
        print("\n" + "="*50)
        print("TRAINING XGBOOST MODEL")
        print("="*50 + "\n")
        
        # Calculate scale_pos_weight for imbalanced data
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        print(f"Class balance:")
        print(f"  Negative samples: {neg_count}")
        print(f"  Positive samples: {pos_count}")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}\n")
        
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning...")
            
            # Define parameter search space
            param_distributions = {
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
            
            # Base model
            base_model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            # Randomized search with F1-score optimization
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=30,  # Number of parameter combinations to try
                scoring='f1',  # Optimize for F1-score
                cv=5,  # 5-fold cross-validation
                verbose=1,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit the model
            random_search.fit(X_train, y_train)
            
            # Best model and parameters
            self.model = random_search.best_estimator_
            self.best_params = random_search.best_params_
            
            print(f"\nBest Parameters: {self.best_params}")
            print(f"Best CV F1-Score: {random_search.best_score_:.4f}")
            
        else:
            # Train with default parameters
            print("Training with default parameters...")
            self.model = xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            self.model.fit(X_train, y_train)
        
        print("\nModel training complete!")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50 + "\n")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}\n")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save confusion matrix plot
        self._plot_confusion_matrix(cm)
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        joblib.dump(metrics, os.path.join(self.model_dir, 'model_metrics.pkl'))
        
        return metrics
    
    def _plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Attrition', 'Attrition'],
                    yticklabels=['No Attrition', 'Attrition'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        os.makedirs(self.model_dir, exist_ok=True)
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'), dpi=150)
        print(f"Confusion matrix saved to {self.model_dir}/confusion_matrix.png")
        plt.close()
    
    def compute_shap_values(self, X_train, X_test):
        """Compute SHAP values for model interpretability"""
        print("\n" + "="*50)
        print("COMPUTING SHAP VALUES")
        print("="*50 + "\n")
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values for test set
        print("Computing SHAP values for test set...")
        self.shap_values = self.shap_explainer.shap_values(X_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(self.shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features (SHAP):")
        print(feature_importance.head(10))
        
        # Save feature importance
        feature_importance.to_csv(
            os.path.join(self.model_dir, 'feature_importance.csv'), 
            index=False
        )
        
        # Save SHAP values
        joblib.dump(self.shap_values, os.path.join(self.model_dir, 'shap_values.pkl'))
        joblib.dump(self.shap_explainer, os.path.join(self.model_dir, 'shap_explainer.pkl'))
        
        # Create SHAP summary plot
        self._plot_shap_summary(X_test)
        
        return feature_importance
    
    def _plot_shap_summary(self, X_test):
        """Plot and save SHAP summary plot"""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
        print(f"SHAP summary plot saved to {self.model_dir}/shap_summary.png")
        plt.close()
    
    def save_model(self):
        """Save trained model"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        model_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save best parameters if available
        if self.best_params:
            joblib.dump(self.best_params, os.path.join(self.model_dir, 'best_params.pkl'))
            print(f"Best parameters saved to {self.model_dir}/best_params.pkl")


def main():
    """Main execution function"""
    # Load processed data
    print("Loading processed data...")
    
    # Note: In a real pipeline, you would load the processed data from data_processing.py
    # For now, we'll process it here
    from data_processing import DataProcessor
    
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.process_pipeline()
    
    # Train model
    trainer = ModelTrainer()
    trainer.train_xgboost(X_train, y_train, hyperparameter_tuning=True)
    
    # Evaluate model
    metrics = trainer.evaluate_model(X_test, y_test)
    
    # Compute SHAP values
    feature_importance = trainer.compute_shap_values(X_train, X_test)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
