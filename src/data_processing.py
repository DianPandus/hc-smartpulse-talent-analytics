"""
Data Processing Module for HC-SmartPulse
Handles data loading, cleaning, encoding, scaling, and train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class DataProcessor:
    """Process IBM HR Attrition dataset for model training"""
    
    def __init__(self, data_path='data/dataset.csv'):
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Attrition'
        
    def load_data(self):
        """Load the dataset"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def clean_data(self):
        """Remove non-informative features"""
        print("Cleaning data...")
        
        # Features to drop (non-informative)
        drop_features = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
        
        # Only drop columns that exist in the dataset
        existing_drop_features = [col for col in drop_features if col in self.df.columns]
        
        self.df = self.df.drop(columns=existing_drop_features)
        print(f"Dropped features: {existing_drop_features}")
        print(f"Remaining columns: {self.df.shape[1]}")
        
        # Check for missing values
        missing = self.df.isnull().sum().sum()
        print(f"Missing values: {missing}")
        
        return self.df
    
    def encode_categorical_features(self):
        """Encode categorical variables using LabelEncoder"""
        print("Encoding categorical features...")
        
        # Identify categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from encoding list
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        # Encode each categorical column
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            print(f"  Encoded: {col} ({len(le.classes_)} unique values)")
        
        # Encode target variable (Attrition: Yes=1, No=0)
        if self.target_column in self.df.columns:
            le_target = LabelEncoder()
            self.df[self.target_column] = le_target.fit_transform(self.df[self.target_column])
            self.label_encoders[self.target_column] = le_target
            print(f"  Encoded target: {self.target_column} -> {le_target.classes_}")
        
        return self.df
    
    def scale_numerical_features(self, X_train, X_test):
        """Scale numerical features using StandardScaler"""
        print("Scaling numerical features...")
        
        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit scaler on training data
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        
        # Transform test data
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        print(f"  Scaled {len(numerical_cols)} numerical features")
        
        return X_train, X_test
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets (stratified)"""
        print("Splitting data...")
        
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Ensure balanced split for imbalanced data
        )
        
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        print(f"  Class distribution in training: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_artifacts(self, model_dir='models'):
        """Save encoders and scaler for later use"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'feature_encoder.pkl'))
        print(f"Saved label encoders to {model_dir}/feature_encoder.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        print(f"Saved scaler to {model_dir}/scaler.pkl")
        
        # Save feature columns
        joblib.dump(self.feature_columns, os.path.join(model_dir, 'feature_columns.pkl'))
        print(f"Saved feature columns to {model_dir}/feature_columns.pkl")
    
    def process_pipeline(self):
        """Execute full data processing pipeline"""
        print("\n" + "="*50)
        print("DATA PROCESSING PIPELINE")
        print("="*50 + "\n")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Encode categorical features
        self.encode_categorical_features()
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = self.split_data()
        
        # Step 5: Scale numerical features
        X_train, X_test = self.scale_numerical_features(X_train, X_test)
        
        # Step 6: Save artifacts
        self.save_artifacts()
        
        print("\n" + "="*50)
        print("DATA PROCESSING COMPLETE")
        print("="*50 + "\n")
        
        return X_train, X_test, y_train, y_test


def main():
    """Main execution function"""
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.process_pipeline()
    
    print("\nData processing completed successfully!")
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")


if __name__ == "__main__":
    main()
