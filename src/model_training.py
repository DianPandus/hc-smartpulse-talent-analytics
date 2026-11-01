"""
Model Training Module for HC-SmartPulse
Implements XGBoost classifier with hyperparameter tuning and SHAP explainability
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score, precision_score,
    recall_score, average_precision_score, roc_auc_score, precision_recall_curve
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
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
        self.threshold = 0.5
        self.cv_pr_auc = None
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
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
            
            # Define parameter search space (shallow trees + lower learning rates suit a small dataset)
            param_distributions = {
                'max_depth': [2, 3, 4, 5],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'n_estimators': [100, 200, 300, 500],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.3]
            }

            # Base model
            base_model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )

            # Optimize PR-AUC: threshold-independent and informative for an
            # imbalanced target (unlike accuracy)
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=40,
                scoring='average_precision',
                cv=self.cv,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )

            # Fit the model
            random_search.fit(X_train, y_train)

            # Best model and parameters
            self.model = random_search.best_estimator_
            self.best_params = random_search.best_params_
            self.cv_pr_auc = random_search.best_score_

            print(f"\nBest Parameters: {self.best_params}")
            print(f"Best CV PR-AUC: {random_search.best_score_:.4f}")
            
        else:
            # Train with default parameters
            print("Training with default parameters...")
            self.model = xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
            self.model.fit(X_train, y_train)
        
        print("\nModel training complete!")
        return self.model

    def tune_threshold(self, X_train, y_train):
        """Pick the decision threshold that maximizes F1 on out-of-fold training predictions.

        The test set is not used here, so the reported test metrics stay unbiased.
        """
        oof_proba = cross_val_predict(
            xgb.XGBClassifier(**self.model.get_params()),
            X_train, y_train, cv=self.cv, method='predict_proba'
        )[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best = np.argmax(f1[:-1])
        self.threshold = float(thresholds[best])

        print(f"\nTuned decision threshold: {self.threshold:.3f} (out-of-fold F1 {f1[best]:.4f})")
        return self.threshold
    
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50 + "\n")
        
        # Predictions at the tuned threshold
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        metrics = self._score(y_test, y_pred, y_pred_proba)
        metrics['threshold'] = self.threshold
        metrics['cv_pr_auc'] = self.cv_pr_auc
        metrics['n_train'] = int(len(y_train))
        metrics['n_test'] = int(len(y_test))

        print(f"Decision threshold: {self.threshold:.3f}")
        for name in ['accuracy', 'precision', 'recall', 'f1_score', 'pr_auc', 'roc_auc']:
            print(f"  {name:10s} {metrics[name]:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        metrics['confusion_matrix'] = cm.tolist()

        # Save confusion matrix plot
        self._plot_confusion_matrix(cm)

        # Baselines put the scores in context: with 16% attrition, always predicting
        # "No" already reaches ~84% accuracy
        metrics['baselines'] = self._evaluate_baselines(X_train, y_train, X_test, y_test)

        joblib.dump(metrics, os.path.join(self.model_dir, 'model_metrics.pkl'))

        return metrics

    @staticmethod
    def _score(y_true, y_pred, y_proba):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'pr_auc': average_precision_score(y_true, y_proba),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }

    def _evaluate_baselines(self, X_train, y_train, X_test, y_test):
        """Score simple reference models on the same split"""
        baselines = {
            'Majority class': DummyClassifier(strategy='most_frequent'),
            'Logistic Regression': LogisticRegression(max_iter=3000, class_weight='balanced')
        }

        results = {}
        print("\nBaselines (threshold 0.5):")
        for name, model in baselines.items():
            # CV score on the training set is a steadier comparison than 47 test positives
            cv_pr_auc = cross_val_score(
                model, X_train, y_train, cv=self.cv, scoring='average_precision'
            ).mean()

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            results[name] = self._score(y_test, (y_proba >= 0.5).astype(int), y_proba)
            results[name]['cv_pr_auc'] = cv_pr_auc
            r = results[name]
            print(f"  {name:20s} acc {r['accuracy']:.3f} | prec {r['precision']:.3f} | "
                  f"rec {r['recall']:.3f} | F1 {r['f1_score']:.3f} | PR-AUC {r['pr_auc']:.3f} | "
                  f"CV PR-AUC {cv_pr_auc:.3f}")

        return results

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
    trainer.tune_threshold(X_train, y_train)
    
    # Evaluate model
    metrics = trainer.evaluate_model(X_train, y_train, X_test, y_test)
    
    # Compute SHAP values
    feature_importance = trainer.compute_shap_values(X_train, X_test)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
