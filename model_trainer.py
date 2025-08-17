"""
Model training module for AQI Predictor.
Trains and evaluates multiple ML models for AQI prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pickle
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Model Explainability
import shap
import lime
import lime.lime_tabular

from utils.config import config
from utils.logger import get_logger
from features.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class ModelTrainer:
    """Train and evaluate machine learning models for AQI prediction."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.config = config
        self.models_config = self.config.get('models', {})
        self.training_config = self.config.get('training', {})
        self.model_registry_config = self.config.get('model_registry', {})
        
        # Model storage
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_scores = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models with configuration."""
        try:
            # Random Forest
            rf_config = self.models_config.get('random_forest', {})
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=rf_config.get('n_estimators', 100),
                max_depth=rf_config.get('max_depth', 10),
                min_samples_split=rf_config.get('min_samples_split', 5),
                min_samples_leaf=rf_config.get('min_samples_leaf', 2),
                random_state=rf_config.get('random_state', 42)
            )
            
            # XGBoost
            xgb_config = self.models_config.get('xgboost', {})
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=xgb_config.get('n_estimators', 100),
                max_depth=xgb_config.get('max_depth', 6),
                learning_rate=xgb_config.get('learning_rate', 0.1),
                subsample=xgb_config.get('subsample', 0.8),
                colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
                random_state=xgb_config.get('random_state', 42)
            )
            
            # Ridge Regression
            self.models['ridge'] = Ridge(
                alpha=1.0,
                random_state=42
            )
            
            # SVR
            self.models['svr'] = SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            features_df: DataFrame with features and target
            
        Returns:
            Tuple of (X, y) for training
        """
        try:
            # Separate features and target
            feature_columns = [col for col in features_df.columns if col not in ['aqi', 'timestamp']]
            X = features_df[feature_columns]
            y = features_df['aqi']
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Create LSTM model for time series prediction.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled LSTM model
        """
        try:
            lstm_config = self.models_config.get('lstm', {})
            
            model = Sequential([
                LSTM(
                    units=lstm_config.get('units', 50),
                    return_sequences=True,
                    input_shape=input_shape
                ),
                Dropout(lstm_config.get('dropout', 0.2)),
                LSTM(
                    units=lstm_config.get('units', 50),
                    return_sequences=False
                ),
                Dropout(lstm_config.get('dropout', 0.2)),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            raise
    
    def train_models(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Train all models and evaluate performance.
        
        Args:
            features_df: DataFrame with features and target
            
        Returns:
            Dictionary with model performance scores
        """
        try:
            # Prepare data
            X, y = self.prepare_data(features_df)
            
            # Split data
            test_size = self.training_config.get('test_size', 0.2)
            validation_size = self.training_config.get('validation_size', 0.1)
            random_state = self.training_config.get('random_state', 42)
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size + validation_size, random_state=random_state
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_size/(test_size + validation_size), 
                random_state=random_state
            )
            
            logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Train and evaluate each model
            results = {}
            
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_val)
                    
                    # Calculate metrics
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
                    
                    results[name] = {
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'mape': mape
                    }
                    
                    self.model_scores[name] = results[name]
                    
                    logger.info(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    results[name] = {
                        'rmse': float('inf'),
                        'mae': float('inf'),
                        'r2': 0.0,
                        'mape': float('inf')
                    }
            
            # Find best model
            self._select_best_model(results)
            
            # Save all trained models
            for name in self.models.keys():
                try:
                    self.save_model(name)
                except Exception as e:
                    logger.warning(f"Could not save model {name}: {e}")
            
            # Save results
            self._save_training_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def _select_best_model(self, results: Dict[str, Dict[str, float]]):
        """Select the best model based on RMSE."""
        try:
            best_rmse = float('inf')
            best_name = None
            
            for name, scores in results.items():
                if scores['rmse'] < best_rmse:
                    best_rmse = scores['rmse']
                    best_name = name
            
            if best_name:
                self.best_model = self.models[best_name]
                self.best_model_name = best_name
                logger.info(f"Best model: {best_name} (RMSE: {best_rmse:.2f})")
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")
    
    def _save_training_results(self, results: Dict[str, Dict[str, float]]):
        """Save training results to file."""
        try:
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'model_scores': results,
                'best_model': self.best_model_name,
                'training_config': self.training_config
            }
            
            results_file = self.models_dir / 'training_results.json'
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Training results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
    
    def save_model(self, model_name: str = 'best') -> str:
        """
        Save trained model to file.
        
        Args:
            model_name: Name of model to save (or 'best' for best model)
            
        Returns:
            Path to saved model file
        """
        try:
            if model_name == 'best':
                if self.best_model is None:
                    raise ValueError("No best model available")
                model = self.best_model
                name = self.best_model_name
            else:
                model = self.models.get(model_name)
                name = model_name
            
            if model is None:
                raise ValueError(f"Model {model_name} not found")
            
            # Save model
            model_file = self.models_dir / f"{name}.pkl"
            joblib.dump(model, model_file)
            
            # Save metadata
            metadata = {
                'model_name': name,
                'timestamp': datetime.now().isoformat(),
                'scores': self.model_scores.get(name, {}),
                'feature_columns': getattr(model, 'feature_names_in_', []).tolist()
            }
            
            metadata_file = self.models_dir / f"{name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model {name} saved to {model_file}")
            return str(model_file)
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            raise
    
    def load_model(self, model_name: str = 'best'):
        """
        Load trained model from file.
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Loaded model
        """
        try:
            model_file = self.models_dir / f"{model_name}.pkl"
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file {model_file} not found")
            
            model = joblib.load(model_file)
            logger.info(f"Model {model_name} loaded from {model_file}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def load_best_model(self):
        """Load the best trained model."""
        try:
            if self.best_model_name:
                return self.load_model(self.best_model_name)
            else:
                # Try to load from saved results
                results_file = self.models_dir / 'training_results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    best_model = results.get('best_model')
                    if best_model:
                        return self.load_model(best_model)
                
                logger.warning("No best model found")
                return None
                
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            return None
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all trained models."""
        return self.model_scores
    
    def get_best_model_name(self) -> Optional[str]:
        """Get the name of the best model."""
        return self.best_model_name
    
    def get_feature_importance(self, model_name: str = 'best') -> Dict[str, float]:
        """
        Get feature importance for a model.
        
        Args:
            model_name: Name of model to analyze
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            if model_name == 'best':
                model = self.best_model
                name = self.best_model_name
            else:
                model = self.models.get(model_name)
                name = model_name
            
            if model is None:
                return {}
            
            # Get feature names
            feature_names = getattr(model, 'feature_names_in_', [])
            if not feature_names:
                return {}
            
            # Get importance scores
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                return {}
            
            # Create importance dictionary
            importance_dict = dict(zip(feature_names, importance))
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {model_name}: {e}")
            return {}
    
    def explain_prediction(self, features: pd.DataFrame, model_name: str = 'best') -> Dict[str, Any]:
        """
        Explain model prediction using SHAP and LIME.
        
        Args:
            features: Feature values for prediction
            model_name: Name of model to explain
            
        Returns:
            Dictionary with explanation results
        """
        try:
            if model_name == 'best':
                model = self.best_model
            else:
                model = self.models.get(model_name)
            
            if model is None:
                return {}
            
            # SHAP explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            
            # LIME explanation
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                features.values,
                feature_names=features.columns.tolist(),
                class_names=['AQI'],
                mode='regression'
            )
            
            lime_exp = lime_explainer.explain_instance(
                features.iloc[0].values,
                model.predict,
                num_features=len(features.columns)
            )
            
            explanation = {
                'shap_values': shap_values.tolist(),
                'lime_explanation': lime_exp.as_list(),
                'feature_names': features.columns.tolist()
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {} 