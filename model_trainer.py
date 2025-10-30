import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, f1_score, precision_score, recall_score
import shap
import joblib
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self):
        # Initialize individual models with optimized parameters
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=12,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=12,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.9,
            scale_pos_weight=1,
            random_state=42,
            eval_metric='logloss'
        )
        self.feature_names = None
        self.explainer = None
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42, k_neighbors=3)
        
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess the data with SMOTE if training (scaling already done by DataProcessor)."""
        if training:
            # X is already scaled by DataProcessor, just apply SMOTE
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            return X_resampled, y_resampled
        else:
            # X is already scaled by DataProcessor
            return X
    
    def train(self, X, y):
        """Train the ensemble model and create SHAP explainer."""
        # NOTE: Data should already be split and preprocessed by app.py
        # X and y here represent the training data only
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Preprocess training data (apply SMOTE)
        X_train_processed, y_train_processed = self.preprocess_data(X, y)
        
        # Convert to numpy arrays and ensure correct data types
        X_train_processed = np.array(X_train_processed, dtype=np.float32)
        y_train_processed = np.array(y_train_processed, dtype=np.int32)
        
        # Train individual models
        self.rf_model.fit(X_train_processed, y_train_processed)
        self.lgb_model.fit(X_train_processed, y_train_processed)
        self.xgb_model.fit(X_train_processed, y_train_processed)
        
        # Create SHAP explainer using Random Forest (most interpretable)
        self.explainer = shap.TreeExplainer(self.rf_model)
        
        # Return empty metrics dict - metrics are computed externally in app.py
        return {
            'auc_score': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'classification_report': ''
        }
    
    def predict(self, X):
        """Make predictions using the ensemble model."""
        # X is already preprocessed by DataProcessor, no need to preprocess again
        
        # Convert to numpy array and ensure correct data type
        X_processed = np.array(X, dtype=np.float32)
        
        # Get predictions from each model
        rf_pred = self.rf_model.predict_proba(X_processed)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X_processed)[:, 1]
        xgb_pred = self.xgb_model.predict_proba(X_processed)[:, 1]
        
        # Ensemble predictions with weighted voting (LightGBM typically performs best)
        ensemble_pred = (0.3 * rf_pred + 0.4 * lgb_pred + 0.3 * xgb_pred)
        
        return ensemble_pred
    
    def get_risk_category(self, probability):
        """Categorize customers into risk levels based on churn probability."""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def get_feature_importance(self):
        """Get feature importance scores from all models."""
        # Get feature importance from each model
        rf_importance = self.rf_model.feature_importances_
        lgb_importance = self.lgb_model.feature_importances_
        xgb_importance = self.xgb_model.feature_importances_
        
        # Average the importance scores
        avg_importance = (rf_importance + lgb_importance + xgb_importance) / 3
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def get_shap_values(self, X):
        """Get SHAP values for feature importance explanation."""
        if self.explainer is None:
            raise ValueError("Model must be trained before getting SHAP values")
        # X is already preprocessed by DataProcessor
        X_processed = np.array(X, dtype=np.float32)
        return self.explainer.shap_values(X_processed)
    
    def save_model(self, path):
        """Save the trained models."""
        models = {
            'rf_model': self.rf_model,
            'lgb_model': self.lgb_model,
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(models, path)
    
    def load_model(self, path):
        """Load trained models."""
        models = joblib.load(path)
        self.rf_model = models['rf_model']
        self.lgb_model = models['lgb_model']
        self.xgb_model = models['xgb_model']
        self.scaler = models['scaler']
        self.feature_names = models['feature_names']
        self.explainer = shap.TreeExplainer(self.rf_model) 