"""
Data preprocessing module for XplainML
Handles data loading, cleaning, encoding, and splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for machine learning workflows
    """
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.one_hot_encoder = None
        self.imputer = None
        self.feature_names = None
        self.target_name = None
        
    def load_data(self, file_path, target_column=None):
        """
        Load data from CSV or Excel files
        
        Args:
            file_path (str): Path to the data file
            target_column (str): Name of the target column
            
        Returns:
            tuple: (features, target) DataFrames
        """
        try:
            # Determine file type and load accordingly
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
            print(f"Data loaded successfully! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Split features and target
            if target_column:
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset")
                
                features = df.drop(columns=[target_column])
                target = df[target_column]
                self.target_name = target_column
            else:
                features = df
                target = None
                
            self.feature_names = list(features.columns)
            return features, target
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, X, strategy='mean', fill_value=None):
        """
        Handle missing values in the dataset
        
        Args:
            X (DataFrame): Input features
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            fill_value: Value to use when strategy is 'constant'
            
        Returns:
            DataFrame: Data with missing values handled
        """
        # Separate numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        X_processed = X.copy()
        
        # Handle numerical missing values
        if len(numerical_cols) > 0 and X[numerical_cols].isnull().any().any():
            if strategy == 'constant':
                num_imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            else:
                num_imputer = SimpleImputer(strategy=strategy)
            
            X_processed[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
            print(f"Handled missing values in numerical columns: {list(numerical_cols)}")
        
        # Handle categorical missing values
        if len(categorical_cols) > 0 and X[categorical_cols].isnull().any().any():
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_processed[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
            print(f"Handled missing values in categorical columns: {list(categorical_cols)}")
        
        return X_processed
    
    def encode_categorical_variables(self, X, encoding_type='auto', max_categories=10):
        """
        Encode categorical variables
        
        Args:
            X (DataFrame): Input features
            encoding_type (str): 'label', 'onehot', or 'auto'
            max_categories (int): Maximum categories for one-hot encoding
            
        Returns:
            DataFrame: Encoded features
        """
        X_encoded = X.copy()
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return X_encoded
        
        for col in categorical_cols:
            unique_values = X[col].nunique()
            
            if encoding_type == 'auto':
                # Use one-hot for low cardinality, label encoding for high cardinality
                if unique_values <= max_categories:
                    use_onehot = True
                else:
                    use_onehot = False
            elif encoding_type == 'onehot':
                use_onehot = True
            else:
                use_onehot = False
            
            if use_onehot:
                # One-hot encoding
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded.drop(columns=[col]), dummies], axis=1)
                print(f"One-hot encoded column: {col} ({unique_values} categories)")
            else:
                # Label encoding
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                print(f"Label encoded column: {col} ({unique_values} categories)")
        
        return X_encoded
    
    def scale_features(self, X, scaling_type='standard'):
        """
        Scale numerical features
        
        Args:
            X (DataFrame): Input features
            scaling_type (str): 'standard', 'minmax', or 'none'
            
        Returns:
            DataFrame: Scaled features
        """
        if scaling_type == 'none':
            return X
        
        X_scaled = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return X_scaled
        
        if scaling_type == 'standard':
            self.scaler = StandardScaler()
        elif scaling_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_type must be 'standard', 'minmax', or 'none'")
        
        X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        print(f"Applied {scaling_type} scaling to numerical columns")
        
        return X_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=None):
        """
        Split data into training and testing sets
        
        Args:
            X (DataFrame): Features
            y (Series): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random seed
            stratify: Whether to stratify split based on target
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Auto-detect if we should stratify for classification
        if stratify is None and y is not None:
            if y.dtype == 'object' or y.nunique() < 20:
                stratify = y
                print("Auto-detected classification task - using stratified split")
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
            print(f"Data split: Train {X_train.shape[0]} samples, Test {X_test.shape[0]} samples")
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            print(f"Data split: Train {X_train.shape[0]} samples, Test {X_test.shape[0]} samples")
            return X_train, X_test, None, None
    
    def preprocess_pipeline(self, file_path, target_column, test_size=0.2, 
                           missing_strategy='mean', encoding_type='auto', 
                           scaling_type='standard', random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path (str): Path to data file
            target_column (str): Target column name
            test_size (float): Test set proportion
            missing_strategy (str): Missing value strategy
            encoding_type (str): Categorical encoding type
            scaling_type (str): Feature scaling type
            random_state (int): Random seed
            
        Returns:
            dict: Preprocessed data splits and metadata
        """
        print("=== Starting Data Preprocessing Pipeline ===")
        
        # Load data
        X, y = self.load_data(file_path, target_column)
        
        # Handle missing values
        X = self.handle_missing_values(X, strategy=missing_strategy)
        
        # Encode categorical variables
        X = self.encode_categorical_variables(X, encoding_type=encoding_type)
        
        # Scale features
        X = self.scale_features(X, scaling_type=scaling_type)
        
        # Split data
        if y is not None:
            X_train, X_test, y_train, y_test = self.split_data(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = self.split_data(
                X, None, test_size=test_size, random_state=random_state
            )
        
        # Prepare return dictionary
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'target_name': target_column,
            'original_shape': X.shape,
            'task_type': self._detect_task_type(y) if y is not None else 'unsupervised'
        }
        
        print("=== Preprocessing Complete ===")
        print(f"Task type: {result['task_type']}")
        print(f"Features: {len(result['feature_names'])}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        return result
    
    def _detect_task_type(self, y):
        """Detect if task is regression or classification"""
        if y is None:
            return 'unsupervised'
        
        if y.dtype == 'object' or y.nunique() < 20:
            return 'classification'
        else:
            return 'regression'
    
    def transform_new_data(self, X_new):
        """
        Transform new data using fitted preprocessors
        
        Args:
            X_new (DataFrame): New data to transform
            
        Returns:
            DataFrame: Transformed data
        """
        X_transformed = X_new.copy()
        
        # Handle missing values (using same imputation as training)
        X_transformed = self.handle_missing_values(X_transformed)
        
        # Apply categorical encoding
        categorical_cols = X_new.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                # Handle unseen categories
                X_transformed[col] = X_transformed[col].astype(str)
                known_categories = self.label_encoders[col].classes_
                X_transformed[col] = X_transformed[col].apply(
                    lambda x: x if x in known_categories else known_categories[0]
                )
                X_transformed[col] = self.label_encoders[col].transform(X_transformed[col])
        
        # Apply scaling
        if self.scaler is not None:
            numerical_cols = X_transformed.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                X_transformed[numerical_cols] = self.scaler.transform(X_transformed[numerical_cols])
        
        return X_transformed


def create_sample_data(filename='sample_data.csv', n_samples=1000):
    """
    Create a sample dataset for testing
    
    Args:
        filename (str): Output filename
        n_samples (int): Number of samples to generate
    """
    np.random.seed(42)
    
    # Generate features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    experience = age - np.random.randint(18, 25, n_samples)
    experience = np.clip(experience, 0, None)
    
    # Create target variable (classification)
    # Higher income, education, and experience increase probability of class 1
    score = (income / 100000 + 
             np.where(education == 'PhD', 3, 
                     np.where(education == 'Master', 2,
                             np.where(education == 'Bachelor', 1, 0))) +
             experience / 20)
    
    prob = 1 / (1 + np.exp(-score + 2))  # Sigmoid function
    target = np.random.binomial(1, prob, n_samples)
    
    # Add some missing values
    income[np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)] = np.nan
    education[np.random.choice(n_samples, size=int(0.03 * n_samples), replace=False)] = None
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'experience': experience,
        'target': target
    })
    
    df.to_csv(filename, index=False)
    print(f"Sample data created: {filename}")
    print(f"Shape: {df.shape}")
    print(df.head())
    
    return df


if __name__ == "__main__":
    # Example usage
    print("XplainML Data Preprocessing Module")
    
    # Create sample data
    sample_df = create_sample_data('data/sample_data.csv')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    result = preprocessor.preprocess_pipeline(
        file_path='data/sample_data.csv',
        target_column='target',
        test_size=0.2,
        missing_strategy='mean',
        encoding_type='auto',
        scaling_type='standard'
    )
    
    print("\nPreprocessing completed successfully!")