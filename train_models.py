"""
Training pipeline for AQI Predictor.
Loads features from feature store and trains all models.
"""

import pandas as pd
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Any

from models.model_trainer import ModelTrainer
from features.feature_pipeline import FeaturePipeline
from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """Training pipeline for AQI prediction."""
    
    def __init__(self):
        """Initialize training pipeline."""
        self.config = config
        self.feature_pipeline = FeaturePipeline()
        self.model_trainer = ModelTrainer()
    
    def load_training_data(self, hours: int = 168) -> pd.DataFrame:
        """
        Load training data from feature store.
        
        Args:
            hours: Number of hours of data to load (default: 1 week)
            
        Returns:
            DataFrame with training features
        """
        logger.info(f"Loading training data from last {hours} hours")
        
        try:
            # Load features from feature store
            features_df = self.feature_pipeline.get_features(hours=hours)
            
            if features_df.empty:
                logger.warning("No features found in feature store. Running feature pipeline...")
                # Run feature pipeline to generate some data
                asyncio.run(self.feature_pipeline.run_pipeline())
                features_df = self.feature_pipeline.get_features(hours=hours)
            
            if features_df.empty:
                logger.warning("Still no features available. Using synthetic data for demonstration...")
                # Create synthetic data for demonstration
                features_df = self._create_synthetic_data()
            
            logger.info(f"Loaded {len(features_df)} training samples")
            return features_df
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """
        Create synthetic training data for demonstration.
        
        Returns:
            DataFrame with synthetic features
        """
        logger.info("Creating synthetic training data")
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        # Time features
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='H'
        )[:n_samples]
        
        # Weather features
        temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 2, n_samples)
        humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 5, n_samples)
        pressure = 1013 + 5 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 1, n_samples)
        wind_speed = 5 + 3 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 1, n_samples)
        
        # Pollutant features
        pm25 = 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 3, n_samples)
        pm10 = 25 + 15 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 5, n_samples)
        o3 = 30 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 5, n_samples)
        no2 = 20 + 15 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 3, n_samples)
        
        # Create DataFrame
        data = {
            'timestamp': timestamps,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'pm25': pm25,
            'pm10': pm10,
            'o3': o3,
            'no2': no2,
            'so2': np.random.normal(5, 2, n_samples),
            'co': np.random.normal(1000, 200, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate AQI (simplified)
        df['aqi'] = np.maximum(
            df['pm25'] * 2.5,
            df['pm10'] * 1.0
        ) + np.random.normal(0, 5, n_samples)
        
        # Ensure AQI is positive
        df['aqi'] = np.maximum(df['aqi'], 0)
        
        logger.info(f"Created synthetic dataset with {len(df)} samples")
        return df
    
    def run_training(self, hours: int = 168) -> Dict[str, Dict[str, float]]:
        """
        Run the complete training pipeline.
        
        Args:
            hours: Number of hours of data to use for training
            
        Returns:
            Dictionary with model scores
        """
        logger.info("Starting training pipeline")
        
        try:
            # 1. Load training data
            features_df = self.load_training_data(hours)
            
            # 2. Train all models
            model_scores = self.model_trainer.train_models(features_df)
            
            # 3. Save training results
            self._save_training_results(model_scores)
            
            # 4. Generate feature importance report
            self._generate_feature_importance_report()
            
            logger.info("Training pipeline completed successfully")
            return model_scores
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def _save_training_results(self, model_scores: Dict[str, Dict[str, float]]):
        """
        Save training results to file.
        
        Args:
            model_scores: Dictionary with model scores
        """
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'model_scores': model_scores,
                'best_model': self.model_trainer.best_model_name,
                'best_score': model_scores[self.model_trainer.best_model_name]['rmse'] if self.model_trainer.best_model_name else None
            }
            
            # Save to file
            results_file = Path('data/training_results.json')
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Training results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
    
    def _generate_feature_importance_report(self):
        """Generate feature importance report."""
        try:
            if self.model_trainer.best_model_name:
                feature_importance = self.model_trainer.get_feature_importance()
                
                # Save feature importance
                importance_file = Path('data/feature_importance.json')
                importance_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(importance_file, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                
                logger.info(f"Feature importance saved to {importance_file}")
                
                # Log top features
                top_features = list(feature_importance.items())[:10]
                logger.info("Top 10 most important features:")
                for feature, importance in top_features:
                    logger.info(f"  {feature}: {importance:.4f}")
            
        except Exception as e:
            logger.error(f"Error generating feature importance report: {e}")
    
    def evaluate_model_performance(self) -> Dict[str, Any]:
        """
        Evaluate model performance and generate reports.
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model performance")
        
        try:
            # Load recent data for evaluation
            recent_features = self.feature_pipeline.get_features(hours=24)
            
            if recent_features.empty:
                logger.warning("No recent data available for evaluation")
                return {}
            
            # Prepare data
            X, y = self.model_trainer.prepare_data(recent_features)
            
            if X.empty or y is None:
                logger.warning("No valid data for evaluation")
                return {}
            
            # Evaluate best model
            if self.model_trainer.best_model and self.model_trainer.best_model_name:
                metrics = self.model_trainer.evaluate_model(
                    self.model_trainer.best_model, X, y, self.model_trainer.best_model_name
                )
                
                evaluation_results = {
                    'timestamp': datetime.now().isoformat(),
                    'model_name': self.model_trainer.best_model_name,
                    'metrics': metrics,
                    'data_points': len(X)
                }
                
                # Save evaluation results
                eval_file = Path('data/evaluation_results.json')
                eval_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(eval_file, 'w') as f:
                    json.dump(evaluation_results, f, indent=2)
                
                logger.info(f"Evaluation results saved to {eval_file}")
                return evaluation_results
            
            return {}
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {}


def main():
    """Main function to run the training pipeline."""
    pipeline = TrainingPipeline()
    
    # Run training
    model_scores = pipeline.run_training()
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    for model_name, scores in model_scores.items():
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE: {scores['rmse']:.2f}")
        print(f"  MAE:  {scores['mae']:.2f}")
        print(f"  R²:   {scores['r2']:.3f}")
    
    print(f"\nBest Model: {pipeline.model_trainer.best_model_name}")
    print("="*50)


if __name__ == "__main__":
    main() 