#!/usr/bin/env python3
"""
Main script to run the AQI Predictor system.
This script can run the feature pipeline, train models, and launch the web application.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from features.feature_pipeline import FeaturePipeline
from models.train_models import TrainingPipeline
from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)


def run_feature_pipeline():
    """Run the feature pipeline."""
    logger.info("Starting feature pipeline...")
    try:
        pipeline = FeaturePipeline()
        success = pipeline.run_pipeline()
        if success:
            logger.info("Feature pipeline completed successfully")
            return True
        else:
            logger.error("Feature pipeline failed")
            return False
    except Exception as e:
        logger.error(f"Feature pipeline failed: {e}")
        return False


def run_training_pipeline():
    """Run the training pipeline."""
    logger.info("Starting training pipeline...")
    try:
        pipeline = TrainingPipeline()
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
        
        print("="*50)
        logger.info("Training pipeline completed successfully")
        return True
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return False


def run_web_app():
    """Run the web application."""
    logger.info("Starting web application...")
    try:
        import subprocess
        import sys
        
        # Run streamlit app
        cmd = [sys.executable, "-m", "streamlit", "run", "src/webapp/app.py"]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        logger.error(f"Web application failed: {e}")
        return False


def run_backfill():
    """Run historical data backfill."""
    logger.info("Starting historical data backfill...")
    try:
        pipeline = FeaturePipeline()
        pipeline.backfill_historical_data(days=30)
        logger.info("Historical data backfill completed successfully")
        return True
    except Exception as e:
        logger.error(f"Historical data backfill failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AQI Predictor System")
    parser.add_argument(
        "action",
        choices=["features", "train", "webapp", "backfill", "all"],
        help="Action to perform"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Load configuration if specified
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    success = True
    
    if args.action == "features":
        success = run_feature_pipeline()
    elif args.action == "train":
        success = run_training_pipeline()
    elif args.action == "webapp":
        success = run_web_app()
    elif args.action == "backfill":
        success = run_backfill()
    elif args.action == "all":
        # Run all components
        logger.info("Running complete AQI Predictor system...")
        
        # 1. Run feature pipeline
        if not run_feature_pipeline():
            logger.error("Feature pipeline failed, stopping")
            return 1
        
        # 2. Run training pipeline
        if not run_training_pipeline():
            logger.error("Training pipeline failed, stopping")
            return 1
        
        # 3. Run web application
        success = run_web_app()
    
    if success:
        logger.info("System completed successfully")
        return 0
    else:
        logger.error("System failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 