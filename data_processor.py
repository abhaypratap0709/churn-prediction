import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None
        self.is_fitted = False
        
    def fit(self, df):
        """Fit preprocessors on training data only."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Remove CustomerID as it's not a predictive feature
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        # Apply feature engineering BEFORE storing columns
        df = self._engineer_features(df)
        
        # Store feature columns for later use (exclude churn)
        self.feature_columns = [col for col in df.columns if col != 'churn']
        
        # Separate churn column if it exists
        churn_col = None
        if 'churn' in df.columns:
            churn_col = df['churn'].copy()
            df = df.drop('churn', axis=1)
        
        # Handle missing values - separate churn column from other numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fit imputer on numeric columns (excluding churn)
        if len(numeric_cols) > 0:
            self.imputer.fit(df[numeric_cols])
        
        # Fit encoders on categorical columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(df[col].astype(str))
        
        # Fit scaler on numeric features (excluding churn)
        if len(numeric_cols) > 0:
            self.scaler.fit(df[numeric_cols])
        
        self.is_fitted = True
    
    def transform(self, df):
        """Transform data using fitted preprocessors."""
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before transform")
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Remove CustomerID as it's not a predictive feature
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        # Apply feature engineering (same as in fit)
        df = self._engineer_features(df)
        
        # Separate churn column if it exists
        churn_col = None
        if 'churn' in df.columns:
            churn_col = df['churn'].copy()
            df = df.drop('churn', axis=1)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer.transform(df[numeric_cols])
        
        # Encode categorical columns
        for col in categorical_cols:
            if col in self.label_encoders:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories by using the most frequent one
                    df[col] = 0
        
        # Scale numeric features
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # Add churn column back if it existed
        if churn_col is not None:
            df['churn'] = churn_col.astype(int)
        
        # Ensure we only return the features that were used during training
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_columns]
            if churn_col is not None:
                df['churn'] = churn_col.astype(int)
        
        return df
        
    def preprocess_data(self, df):
        """Preprocess the input dataframe with enhanced feature engineering (deprecated - use fit/transform)."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Remove CustomerID as it's not a predictive feature
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        # Feature Engineering
        df = self._engineer_features(df)
        
        # Handle missing values - separate churn column from other numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Store feature columns for later use (exclude churn)
        self.feature_columns = [col for col in df.columns if col != 'churn']
        
        # Separate churn column if it exists
        churn_col = None
        if 'churn' in df.columns:
            churn_col = df['churn'].copy()
            df = df.drop('churn', axis=1)
            # Recalculate columns after dropping churn
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns (excluding churn)
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Encode categorical columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Scale numeric features (excluding churn)
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        # Add churn column back if it existed
        if churn_col is not None:
            df['churn'] = churn_col.astype(int)
        
        return df
    
    def _engineer_features(self, df):
        """Create additional features for better prediction (no data leakage)."""
        # Create customer value score (spend per month)
        if 'Total Spend' in df.columns and 'Tenure' in df.columns:
            df['Spend_Per_Month'] = df['Total Spend'] / (df['Tenure'] + 1)  # +1 to avoid division by zero
        
        # Create support ratio (calls per month)
        if 'Support Calls' in df.columns and 'Tenure' in df.columns:
            df['Support_Ratio'] = df['Support Calls'] / (df['Tenure'] + 1)
        
        # Create usage efficiency (usage frequency vs tenure)
        if 'Usage Frequency' in df.columns and 'Tenure' in df.columns:
            df['Usage_Efficiency'] = df['Usage Frequency'] / (df['Tenure'] + 1)
        
        # Create subscription value indicator
        if 'Subscription Type' in df.columns and 'Total Spend' in df.columns:
            subscription_values = {'Basic': 1, 'Standard': 2, 'Premium': 3}
            df['Subscription_Value'] = df['Subscription Type'].map(subscription_values)
            df['Spend_vs_Subscription'] = df['Total Spend'] / df['Subscription_Value']
        
        # Create contract stability indicator
        if 'Contract Length' in df.columns:
            contract_stability = {'Monthly': 1, 'Quarterly': 3, 'Annual': 12}
            df['Contract_Stability'] = df['Contract Length'].map(contract_stability)
        
        # NOTE: Removed Risk_Score feature that used .max() on full dataset - this was data leakage
        # Using .max() includes test data statistics during training
        
        return df
    
    def preprocess_single_row(self, row):
        """Preprocess a single row of data for real-time prediction."""
        # Convert to DataFrame if it's not already
        if isinstance(row, pd.Series):
            row = pd.DataFrame([row])
        
        # Remove CustomerID if present
        if 'CustomerID' in row.columns:
            row = row.drop('CustomerID', axis=1)
        
        # Apply feature engineering
        row = self._engineer_features(row)
        
        # Handle missing values
        numeric_cols = row.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = row.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            row[numeric_cols] = self.imputer.transform(row[numeric_cols])
        
        # Encode categorical columns
        for col in categorical_cols:
            if col in self.label_encoders:
                # Handle unseen categories by using the most frequent one
                try:
                    row[col] = self.label_encoders[col].transform(row[col].astype(str))
                except ValueError:
                    # If category not seen during training, use 0 (first encoded value)
                    row[col] = 0
        
        # Scale numeric features
        if len(numeric_cols) > 0:
            row[numeric_cols] = self.scaler.transform(row[numeric_cols])
        
        # Ensure we only return the features that were used during training
        if self.feature_columns:
            # Add any missing columns with default values
            for col in self.feature_columns:
                if col not in row.columns:
                    row[col] = 0
            row = row[self.feature_columns]
        
        return row 