"""
Feature engineering module for AQI Predictor.
Creates time-based features, derived features, and prepares data for ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    """Feature engineering for AQI prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.config = config
        self.features_config = self.config.get('features', {})
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with time features
        """
        try:
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                df['timestamp'] = pd.to_datetime(datetime.now())
            
            # Basic time features
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
            df['season'] = ((df['month'] % 12 + 3) // 3).astype(int)
            
            # Cyclical features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            logger.info("Time features created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            raise
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with weather features
        """
        try:
            weather_features = self.features_config.get('weather_features', [])
            
            # Temperature features
            if 'temperature' in df.columns:
                df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
                df['temperature_squared'] = df['temperature'] ** 2
                df['temperature_cubed'] = df['temperature'] ** 3
            
            # Humidity features
            if 'humidity' in df.columns:
                df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
                df['humidity_squared'] = df['humidity'] ** 2
            
            # Pressure features
            if 'pressure' in df.columns:
                df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce')
                df['pressure_normalized'] = (df['pressure'] - 1013.25) / 1013.25
            
            # Wind features
            if 'wind_speed' in df.columns:
                df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
                df['wind_speed_squared'] = df['wind_speed'] ** 2
                
            if 'wind_direction' in df.columns:
                df['wind_direction'] = pd.to_numeric(df['wind_direction'], errors='coerce')
                df['wind_direction_sin'] = np.sin(np.radians(df['wind_direction']))
                df['wind_direction_cos'] = np.cos(np.radians(df['wind_direction']))
            
            # Visibility features
            if 'visibility' in df.columns:
                df['visibility'] = pd.to_numeric(df['visibility'], errors='coerce')
                df['visibility_normalized'] = df['visibility'] / 10000  # Normalize to 0-1
            
            # Cloud cover features
            if 'clouds' in df.columns:
                df['clouds'] = pd.to_numeric(df['clouds'], errors='coerce')
                df['clouds_normalized'] = df['clouds'] / 100  # Normalize to 0-1
            
            logger.info("Weather features created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating weather features: {e}")
            raise
    
    def create_pollutant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create pollutant-related features.
        
        Args:
            df: DataFrame with pollutant data
            
        Returns:
            DataFrame with pollutant features
        """
        try:
            pollutant_features = self.features_config.get('pollutant_features', [])
            
            # PM2.5 features
            if 'pm25' in df.columns:
                df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
                df['pm25_squared'] = df['pm25'] ** 2
                df['pm25_log'] = np.log1p(df['pm25'])  # log(1+x) to handle zeros
            
            # PM10 features
            if 'pm10' in df.columns:
                df['pm10'] = pd.to_numeric(df['pm10'], errors='coerce')
                df['pm10_squared'] = df['pm10'] ** 2
                df['pm10_log'] = np.log1p(df['pm10'])
            
            # Ozone features
            if 'o3' in df.columns:
                df['o3'] = pd.to_numeric(df['o3'], errors='coerce')
                df['o3_squared'] = df['o3'] ** 2
                df['o3_log'] = np.log1p(df['o3'])
            
            # NO2 features
            if 'no2' in df.columns:
                df['no2'] = pd.to_numeric(df['no2'], errors='coerce')
                df['no2_squared'] = df['no2'] ** 2
                df['no2_log'] = np.log1p(df['no2'])
            
            # SO2 features
            if 'so2' in df.columns:
                df['so2'] = pd.to_numeric(df['so2'], errors='coerce')
                df['so2_squared'] = df['so2'] ** 2
                df['so2_log'] = np.log1p(df['so2'])
            
            # CO features
            if 'co' in df.columns:
                df['co'] = pd.to_numeric(df['co'], errors='coerce')
                df['co_squared'] = df['co'] ** 2
                df['co_log'] = np.log1p(df['co'])
            
            # Combined pollutant features
            pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
            available_cols = [col for col in pollutant_cols if col in df.columns]
            
            if len(available_cols) >= 2:
                df['pollutant_sum'] = df[available_cols].sum(axis=1)
                df['pollutant_mean'] = df[available_cols].mean(axis=1)
                df['pollutant_std'] = df[available_cols].std(axis=1)
                df['pollutant_max'] = df[available_cols].max(axis=1)
            
            logger.info("Pollutant features created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating pollutant features: {e}")
            raise
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features (lag, rolling stats, change rates).
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with derived features
        """
        try:
            derived_features = self.features_config.get('derived_features', [])
            
            # Sort by timestamp for lag features
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Lag features
            lag_features = ['aqi', 'temperature', 'humidity', 'wind_speed', 'pm25', 'pm10']
            for feature in lag_features:
                if feature in df.columns:
                    df[f'{feature}_lag_1'] = df[feature].shift(1)
                    df[f'{feature}_lag_2'] = df[feature].shift(2)
                    df[f'{feature}_lag_3'] = df[feature].shift(3)
            
            # Change rate features
            for feature in lag_features:
                if feature in df.columns:
                    df[f'{feature}_change_rate'] = df[feature].diff()
                    df[f'{feature}_change_rate_pct'] = df[feature].pct_change()
            
            # Rolling statistics (24-hour windows)
            for feature in lag_features:
                if feature in df.columns:
                    df[f'{feature}_rolling_mean_24h'] = df[feature].rolling(window=24, min_periods=1).mean()
                    df[f'{feature}_rolling_std_24h'] = df[feature].rolling(window=24, min_periods=1).std()
                    df[f'{feature}_rolling_min_24h'] = df[feature].rolling(window=24, min_periods=1).min()
                    df[f'{feature}_rolling_max_24h'] = df[feature].rolling(window=24, min_periods=1).max()
            
            # Rolling statistics (7-day windows)
            for feature in lag_features:
                if feature in df.columns:
                    df[f'{feature}_rolling_mean_7d'] = df[feature].rolling(window=168, min_periods=1).mean()  # 7*24 hours
                    df[f'{feature}_rolling_std_7d'] = df[feature].rolling(window=168, min_periods=1).std()
            
            # Interaction features
            if 'temperature' in df.columns and 'humidity' in df.columns:
                df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            
            if 'pm25' in df.columns and 'pm10' in df.columns:
                df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1e-8)  # Avoid division by zero
            
            if 'wind_speed' in df.columns and 'pm25' in df.columns:
                df['wind_pm25_interaction'] = df['wind_speed'] * df['pm25']
            
            logger.info("Derived features created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating derived features: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Fill missing values
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    # Use forward fill first, then backward fill, then mean
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(df[col].mean())
            
            # For categorical/text columns, fill with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
            
            logger.info("Missing values handled successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
    
    def flatten_nested_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten nested data structure from data collector.
        
        Args:
            df: DataFrame with nested weather and air quality data
            
        Returns:
            Flattened DataFrame with extracted features
        """
        try:
            flattened_df = df.copy()
            
            # Extract weather features from nested structure
            if 'weather' in flattened_df.columns:
                weather_data = flattened_df['weather'].apply(
                    lambda x: x if isinstance(x, dict) else {}
                )
                
                # Extract individual weather fields
                weather_fields = ['temperature', 'feels_like', 'humidity', 'pressure', 
                                'wind_speed', 'wind_direction', 'visibility', 'clouds']
                
                for field in weather_fields:
                    if field in weather_data.iloc[0] if len(weather_data) > 0 else {}:
                        flattened_df[field] = weather_data.apply(
                            lambda x: x.get(field, None) if isinstance(x, dict) else None
                        )
            
            # Extract air pollution features from nested structure
            if 'air_pollution_openweather' in flattened_df.columns:
                air_pollution_data = flattened_df['air_pollution_openweather'].apply(
                    lambda x: x if isinstance(x, dict) else {}
                )
                
                # Extract individual pollutant fields
                pollutant_fields = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
                
                for field in pollutant_fields:
                    if field in air_pollution_data.iloc[0] if len(air_pollution_data) > 0 else {}:
                        flattened_df[field] = air_pollution_data.apply(
                            lambda x: x.get(field, None) if isinstance(x, dict) else None
                        )
                
                # Extract AQI if available
                if 'aqi' in air_pollution_data.iloc[0] if len(air_pollution_data) > 0 else {}:
                    flattened_df['aqi_openweather'] = air_pollution_data.apply(
                        lambda x: x.get('aqi', None) if isinstance(x, dict) else None
                    )
            
            # Extract AQICN features if available
            if 'air_quality_aqicn' in flattened_df.columns:
                aqicn_data = flattened_df['air_quality_aqicn'].apply(
                    lambda x: x if isinstance(x, dict) else {}
                )
                
                # Extract AQI if available
                if 'aqi' in aqicn_data.iloc[0] if len(aqicn_data) > 0 else {}:
                    flattened_df['aqi_aqicn'] = aqicn_data.apply(
                        lambda x: x.get('aqi', None) if isinstance(x, dict) else None
                    )
            
            # Create timestamp if not present
            if 'timestamp' not in flattened_df.columns:
                flattened_df['timestamp'] = pd.Timestamp.now()
            
            logger.info("Data flattened successfully")
            return flattened_df
            
        except Exception as e:
            logger.error(f"Error flattening nested data: {e}")
            raise

    def create_all_features(self, df: pd.DataFrame, target_col: str = 'aqi') -> pd.DataFrame:
        """
        Create all features from raw data.
        
        Args:
            df: DataFrame with raw data
            target_col: Name of target column
            
        Returns:
            DataFrame with all features and target
        """
        try:
            logger.info("Starting feature engineering...")
            
            # Create a copy to avoid modifying original
            features_df = df.copy()
            
            # Flatten nested data structure first
            features_df = self.flatten_nested_data(features_df)
            
            # Create time features
            features_df = self.create_time_features(features_df)
            
            # Create weather features
            features_df = self.create_weather_features(features_df)
            
            # Create pollutant features
            features_df = self.create_pollutant_features(features_df)
            
            # Create derived features
            features_df = self.create_derived_features(features_df)
            
            # Handle missing values
            features_df = self.handle_missing_values(features_df)
            
            # Ensure target column exists
            if target_col not in features_df.columns:
                # Try to find AQI data in flattened structure
                if 'aqi_openweather' in features_df.columns:
                    features_df[target_col] = features_df['aqi_openweather']
                elif 'aqi_aqicn' in features_df.columns:
                    features_df[target_col] = features_df['aqi_aqicn']
                else:
                    # Create synthetic target for demonstration
                    features_df[target_col] = 50 + np.random.normal(0, 10, len(features_df))
            
            # Remove non-feature columns
            exclude_cols = ['timestamp', 'location', 'weather', 'air_pollution_openweather', 
                           'air_quality_aqicn', 'source', 'aqi_openweather', 'aqi_aqicn']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols and col != target_col]
            
            # Ensure we have the target column
            if target_col not in features_df.columns:
                features_df[target_col] = 50 + np.random.normal(0, 10, len(features_df))
            
            # Select only feature columns and target
            final_cols = feature_cols + [target_col]
            final_df = features_df[final_cols]
            
            logger.info(f"Feature engineering completed. Final shape: {final_df.shape}")
            return final_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
    
    def get_feature_columns(self, df: pd.DataFrame, target_col: str = 'aqi') -> List[str]:
        """
        Get list of feature columns (excluding target).
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
            
        Returns:
            List of feature column names
        """
        return [col for col in df.columns if col != target_col]
    
    def prepare_features_and_target(self, df: pd.DataFrame, target_col: str = 'aqi') -> tuple:
        """
        Prepare features and target for training.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame")
            
            # Separate features and target
            feature_cols = self.get_feature_columns(df, target_col)
            features_df = df[feature_cols]
            target_series = df[target_col]
            
            # Handle missing values
            features_df = self.handle_missing_values(features_df)
            target_series = target_series.fillna(target_series.mean())
            
            logger.info(f"Features and target prepared: {features_df.shape[0]} samples, {features_df.shape[1]} features")
            return features_df, target_series
            
        except Exception as e:
            logger.error(f"Error preparing features and target: {e}")
            raise 