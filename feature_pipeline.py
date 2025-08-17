"""
Feature pipeline for AQI Predictor.
Orchestrates data collection, feature engineering, and storage.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pickle

from .data_collector import DataCollector
from .feature_engineering import FeatureEngineer
from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)

class FeaturePipeline:
    """Feature pipeline for collecting, processing, and storing AQI data."""
    
    def __init__(self):
        """Initialize the feature pipeline."""
        self.config = config
        self.feature_store_config = self.config.get('feature_store', {})
        
        # Initialize components
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        
        # Setup storage
        self.db_path = self._setup_storage()
        self._create_tables()
    
    def _setup_storage(self) -> str:
        """Setup feature store storage."""
        try:
            if self.feature_store_config.get('type') == 'local':
                db_path = self.feature_store_config.get('local', {}).get('path', 'data/feature_store.db')
            else:
                db_path = 'data/feature_store.db'
            
            # Create directory if it doesn't exist
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Feature store setup at: {db_path}")
            return db_path
            
        except Exception as e:
            logger.error(f"Error setting up storage: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Raw data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    location TEXT NOT NULL,
                    weather_data TEXT,
                    air_pollution_data TEXT,
                    aqicn_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Features table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    features TEXT NOT NULL,
                    target REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_data_timestamp ON raw_data(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def collect_current_data(self) -> Optional[Dict[str, Any]]:
        """
        Collect current data from APIs.
        
        Returns:
            Current data dictionary or None if failed
        """
        try:
            logger.info("Collecting current data...")
            
            # Collect data
            data = self.data_collector.collect_all_data()
            
            if data:
                logger.info("Current data collected successfully")
                return data
            else:
                logger.warning("No data collected")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting current data: {e}")
            return None
    
    def store_raw_data(self, data: Dict[str, Any]):
        """
        Store raw data in the feature store.

        Args:
            data: Raw data to store
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                else:
                    return obj

            weather_data = convert_datetime(data.get('weather', {}))
            air_pollution_data = convert_datetime(data.get('air_pollution_openweather', {}))
            aqicn_data = convert_datetime(data.get('air_quality_aqicn', {}))

            cursor.execute('''
                INSERT INTO raw_data (timestamp, location, weather_data, air_pollution_data, aqicn_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                data['timestamp'].isoformat() if hasattr(data['timestamp'], 'isoformat') else str(data['timestamp']),
                json.dumps(data['location']),
                json.dumps(weather_data),
                json.dumps(air_pollution_data),
                json.dumps(aqicn_data)
            ))

            conn.commit()
            conn.close()
            logger.info("Raw data stored successfully")

        except Exception as e:
            logger.error(f"Error storing raw data: {e}")
            raise
    
    def get_raw_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Get raw data from the feature store.
        
        Args:
            hours: Number of hours to retrieve
            
        Returns:
            DataFrame with raw data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get data from last N hours
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = '''
                SELECT timestamp, location, weather_data, air_pollution_data, aqicn_data
                FROM raw_data
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(cutoff_time.isoformat(),))
            conn.close()
            
            # Parse JSON columns
            if not df.empty:
                df['location'] = df['location'].apply(json.loads)
                df['weather_data'] = df['weather_data'].apply(json.loads)
                df['air_pollution_data'] = df['air_pollution_data'].apply(json.loads)
                df['aqicn_data'] = df['aqicn_data'].apply(json.loads)
            
            logger.info(f"Retrieved {len(df)} raw data records")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving raw data: {e}")
            return pd.DataFrame()
    
    def create_features_from_raw_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw data.
        
        Args:
            raw_data: Raw data DataFrame
            
        Returns:
            DataFrame with features
        """
        try:
            if raw_data.empty:
                logger.warning("No raw data provided for feature creation")
                return pd.DataFrame()
            
            # Flatten the data structure
            features_list = []
            
            for _, row in raw_data.iterrows():
                # Extract weather data
                weather = row['weather_data']
                air_pollution = row['air_pollution_data']
                aqicn = row['aqicn_data']
                
                # Create feature row
                feature_row = {
                    'timestamp': pd.to_datetime(row['timestamp']),
                    'temperature': weather.get('temperature', 0),
                    'humidity': weather.get('humidity', 0),
                    'pressure': weather.get('pressure', 0),
                    'wind_speed': weather.get('wind_speed', 0),
                    'wind_direction': weather.get('wind_direction', 0),
                    'visibility': weather.get('visibility', 0),
                    'clouds': weather.get('clouds', 0),
                    'aqi': air_pollution.get('aqi', 0),
                    'pm25': air_pollution.get('pm25', 0),
                    'pm10': air_pollution.get('pm10', 0),
                    'o3': air_pollution.get('o3', 0),
                    'no2': air_pollution.get('no2', 0),
                    'so2': air_pollution.get('so2', 0),
                    'co': air_pollution.get('co', 0)
                }
                
                features_list.append(feature_row)
            
            features_df = pd.DataFrame(features_list)
            
            # Create engineered features
            if not features_df.empty:
                features_df = self.feature_engineer.create_all_features(features_df, target_col='aqi')
            
            logger.info(f"Created features from {len(raw_data)} raw data records")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating features from raw data: {e}")
            return pd.DataFrame()
    
    def store_features(self, features_df: pd.DataFrame):
        """
        Store features in the feature store.
        
        Args:
            features_df: DataFrame with features and target
        """
        try:
            if features_df.empty:
                logger.warning("No features to store")
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for _, row in features_df.iterrows():
                try:
                    # Convert features to dictionary (excluding target)
                    features_dict = {}
                    target_value = None
                    
                    for col in features_df.columns:
                        if col == 'aqi':  # Use 'aqi' as target column
                            target_value = row[col]
                        else:
                            # Handle different data types
                            value = row[col]
                            if pd.isna(value):
                                value = 0
                            elif hasattr(value, 'item'):  # Handle numpy types
                                value = value.item()
                            features_dict[col] = value
                    
                    # Ensure we have a target value
                    if target_value is None:
                        target_value = 50  # Default AQI value
                    elif hasattr(target_value, 'item'):
                        target_value = target_value.item()
                    
                    # Create a timestamp if not present
                    timestamp = datetime.now()
                    
                    cursor.execute('''
                        INSERT INTO features (timestamp, features, target)
                        VALUES (?, ?, ?)
                    ''', (
                        timestamp.isoformat(),
                        json.dumps(features_dict),
                        target_value
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error processing feature row: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(features_df)} feature records")
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            raise
    
    def get_features(self, hours: int = 168) -> pd.DataFrame:
        """
        Get features from the feature store.
        
        Args:
            hours: Number of hours to retrieve (default: 7 days)
            
        Returns:
            DataFrame with features
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get data from last N hours
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = '''
                SELECT timestamp, features, target
                FROM features
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(cutoff_time.isoformat(),))
            conn.close()
            
            if df.empty:
                logger.info("No features found in database")
                return df
            
            # Parse JSON features back into DataFrame
            parsed_features = []
            for _, row in df.iterrows():
                try:
                    features_dict = json.loads(row['features'])
                    features_dict['timestamp'] = row['timestamp']
                    features_dict['aqi'] = row['target']  # Add target back
                    parsed_features.append(features_dict)
                except Exception as e:
                    logger.warning(f"Error parsing features row: {e}")
                    continue
            
            if parsed_features:
                # Convert to DataFrame
                result_df = pd.DataFrame(parsed_features)
                
                # Convert timestamp to datetime
                if 'timestamp' in result_df.columns:
                    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
                
                # Ensure numeric columns are properly typed
                numeric_columns = ['aqi', 'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'season',
                                 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                                 'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
                                 'precipitation', 'visibility', 'cloud_cover', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
                
                for col in numeric_columns:
                    if col in result_df.columns:
                        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                
                logger.info(f"Retrieved and parsed {len(result_df)} feature records")
                return result_df
            else:
                logger.warning("No valid features could be parsed")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            return pd.DataFrame()
    
    def backfill_historical_data(self, days: int = 7) -> bool:
        """
        Backfill historical data for training.
        
        Args:
            days: Number of days to backfill
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting historical data backfill for {days} days...")
            
            # Create historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            historical_data = self.data_collector.collect_historical_data(start_date, end_date)
            
            if historical_data:
                # Store raw data first
                for data_point in historical_data:
                    self.store_raw_data(data_point)
                
                # Convert to DataFrame and let feature engineering handle the structure
                historical_df = pd.DataFrame(historical_data)
                
                # Apply feature engineering directly (it handles the nested structure)
                features_df = self.feature_engineer.create_all_features(historical_df, target_col='aqi')
                
                if not features_df.empty:
                    self.store_features(features_df)
                    logger.info(f"Historical data backfill completed: {len(features_df)} feature records created")
                    return True
                else:
                    logger.warning("No features created from historical data")
                    return False
            else:
                logger.warning("No historical data generated")
                return False
                
        except Exception as e:
            logger.error(f"Error during historical data backfill: {e}")
            raise
    
    def run_pipeline(self) -> bool:
        """
        Run the complete feature pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting feature pipeline...")
            
            # Step 1: Collect current data
            current_data = self.collect_current_data()
            if not current_data:
                logger.error("Failed to collect current data")
                return False
            
            # Step 2: Store raw data
            self.store_raw_data(current_data)
            
            # Step 3: Create features
            raw_data = self.get_raw_data(hours=1)  # Get recent data
            features_df = self.create_features_from_raw_data(raw_data)
            
            if not features_df.empty:
                # Step 4: Store features
                self.store_features(features_df)
                
                logger.info("Feature pipeline completed successfully")
                return True
            else:
                logger.warning("No features created")
                return False
                
        except Exception as e:
            logger.error(f"Error running feature pipeline: {e}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the status of the feature pipeline.
        
        Returns:
            Dictionary with pipeline status information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute('SELECT COUNT(*) FROM raw_data')
            raw_data_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM features')
            features_count = cursor.fetchone()[0]
            
            # Get latest timestamps
            cursor.execute('SELECT MAX(timestamp) FROM raw_data')
            latest_raw = cursor.fetchone()[0]
            
            cursor.execute('SELECT MAX(timestamp) FROM features')
            latest_features = cursor.fetchone()[0]
            
            conn.close()
            
            status = {
                'raw_data_count': raw_data_count,
                'features_count': features_count,
                'latest_raw_data': latest_raw,
                'latest_features': latest_features,
                'location': self.config.get_location(),
                'pipeline_status': 'active'
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {
                'raw_data_count': 0,
                'features_count': 0,
                'latest_raw_data': None,
                'latest_features': None,
                'location': self.config.get_location(),
                'pipeline_status': 'error'
            } 