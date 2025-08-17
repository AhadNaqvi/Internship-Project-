"""
Streamlit web application for AQI Predictor.
Provides real-time AQI predictions, historical data visualization, and model explainability.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_pipeline import FeaturePipeline
from models.model_trainer import ModelTrainer
from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)

class AQIPredictorApp:
    """Streamlit web application for AQI Predictor."""
    
    def __init__(self):
        self.feature_pipeline = FeaturePipeline()
        self.model_trainer = ModelTrainer()
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="AQI Predictor",
            page_icon="🌤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .location-info {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def location_selector(self):
        """Location selection sidebar."""
        st.sidebar.header("🌍 Location Selection")
        
        # Get available countries
        countries = config.get_available_countries()
        country_names = [f"{country['name']} ({country['code']})" for country in countries]
        country_codes = [country['code'] for country in countries]
        
        # Country dropdown
        selected_country_display = st.sidebar.selectbox(
            "Select Country:",
            country_names,
            index=country_codes.index(config.get_location().get('country', 'US'))
        )
        
        # Extract country code
        selected_country_code = selected_country_display.split('(')[-1].rstrip(')')
        
        # Get cities for selected country
        cities = config.get_available_cities(selected_country_code)
        city_names = [city['name'] for city in cities]
        
        if city_names:
            # City dropdown
            current_city = config.get_location().get('city', 'New York')
            selected_city = st.sidebar.selectbox(
                "Select City:",
                city_names,
                index=city_names.index(current_city) if current_city in city_names else 0
            )
            
            # Update location button
            if st.sidebar.button("🔄 Update Location"):
                if config.update_location(selected_country_code, selected_city):
                    st.sidebar.success(f"✅ Location updated to {selected_city}, {selected_country_code}")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to update location")
            
            # Show current location info
            city_info = config.get_city_info(selected_country_code, selected_city)
            if city_info:
                st.sidebar.markdown("### 📍 Current Location")
                st.sidebar.markdown(f"""
                **City:** {city_info['name']}  
                **Country:** {city_info['country_name']}  
                **Coordinates:** {city_info['latitude']:.4f}, {city_info['longitude']:.4f}  
                **Timezone:** {city_info['timezone']}
                """)
        else:
            st.sidebar.warning("No cities available for selected country")
    
    def main_header(self):
        """Display main header."""
        st.markdown('<h1 class="main-header">🌤️ AQI Predictor</h1>', unsafe_allow_html=True)
        st.markdown("### Predict Air Quality Index for the next 3 days")
    
    def current_aqi_section(self):
        """Display current AQI information."""
        st.header("📊 Current Air Quality")
        
        try:
            # Collect current data
            current_data = self.feature_pipeline.collect_current_data()
            
            if current_data:
                # Extract AQI data
                aqi_data = current_data.get('air_pollution_openweather', {})
                weather_data = current_data.get('weather', {})
                
                # Create metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    aqi_value = aqi_data.get('aqi', 'N/A')
                    aqi_category = self.get_aqi_category(aqi_data.get('aqi', 0))
                    st.metric(
                        label="Current AQI",
                        value=aqi_value,
                        delta=aqi_category
                    )
                
                with col2:
                    temp = weather_data.get('temperature', 'N/A')
                    if isinstance(temp, (int, float)):
                        temp = f"{temp:.1f}°C"
                    st.metric(
                        label="Temperature",
                        value=temp
                    )
                
                with col3:
                    humidity = weather_data.get('humidity', 'N/A')
                    if isinstance(humidity, (int, float)):
                        humidity = f"{humidity}%"
                    st.metric(
                        label="Humidity",
                        value=humidity
                    )
                
                with col4:
                    wind_speed = weather_data.get('wind_speed', 'N/A')
                    if isinstance(wind_speed, (int, float)):
                        wind_speed = f"{wind_speed} m/s"
                    st.metric(
                        label="Wind Speed",
                        value=wind_speed
                    )
                
                # Detailed pollutant information
                st.subheader("🔬 Pollutant Levels")
                pollutant_cols = st.columns(5)
                
                pollutants = [
                    ('PM2.5', aqi_data.get('pm25', 'N/A'), 'μg/m³'),
                    ('PM10', aqi_data.get('pm10', 'N/A'), 'μg/m³'),
                    ('O₃', aqi_data.get('o3', 'N/A'), 'μg/m³'),
                    ('NO₂', aqi_data.get('no2', 'N/A'), 'μg/m³'),
                    ('SO₂', aqi_data.get('so2', 'N/A'), 'μg/m³')
                ]
                
                for i, (name, value, unit) in enumerate(pollutants):
                    with pollutant_cols[i]:
                        if isinstance(value, (int, float)):
                            value = f"{value:.1f}"
                        st.metric(label=name, value=f"{value} {unit}")
                
            else:
                st.warning("Unable to fetch current AQI data. Please check your API configuration.")
                
        except Exception as e:
            st.error(f"Error fetching current AQI data: {e}")
            logger.error(f"Error in current_aqi_section: {e}")
    
    def predictions_section(self):
        """Display AQI predictions."""
        st.header("🔮 3-Day AQI Forecast")
        
        try:
            # Load trained model
            model = self.model_trainer.load_best_model()
            
            if model is not None:
                # Generate predictions for next 3 days
                predictions = self.generate_predictions(model, days=3)
                
                if predictions:
                    # Create prediction chart
                    fig = go.Figure()
                    
                    dates = [pred['date'] for pred in predictions]
                    aqi_values = [pred['aqi'] for pred in predictions]
                    categories = [pred['category'] for pred in predictions]
                    
                    # Color mapping for AQI categories
                    colors = {
                        'Good': '#00E400',
                        'Moderate': '#FFFF00',
                        'Unhealthy for Sensitive Groups': '#FF7E00',
                        'Unhealthy': '#FF0000',
                        'Very Unhealthy': '#8F3F97',
                        'Hazardous': '#7E0023'
                    }
                    
                    color_values = [colors.get(cat, '#808080') for cat in categories]
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=aqi_values,
                        mode='lines+markers',
                        name='Predicted AQI',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8, color=color_values)
                    ))
                    
                    fig.update_layout(
                        title="AQI Forecast (Next 3 Days)",
                        xaxis_title="Date",
                        yaxis_title="AQI",
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction table
                    st.subheader("📋 Detailed Predictions")
                    pred_df = pd.DataFrame(predictions)
                    pred_df['Date'] = pd.to_datetime(pred_df['date']).dt.strftime('%Y-%m-%d %H:%M')
                    pred_df = pred_df[['Date', 'aqi', 'category', 'confidence']]
                    pred_df.columns = ['Date', 'AQI', 'Category', 'Confidence']
                    
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Health recommendations
                    st.subheader("💡 Health Recommendations")
                    self.display_health_recommendations(predictions)
                    
                else:
                    st.warning("Unable to generate predictions. Please ensure you have sufficient historical data.")
            else:
                st.warning("No trained model found. Please run the training pipeline first.")
                
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            logger.error(f"Error in predictions_section: {e}")
    
    def historical_data_section(self):
        """Display historical AQI data."""
        st.header("📈 Historical AQI Data")
        
        try:
            # Get historical features
            features_df = self.feature_pipeline.get_features()
            
            if not features_df.empty:
                # Data is already parsed, just need to sort by timestamp
                hist_df = features_df.sort_values('timestamp').copy()
                
                # Display basic statistics
                st.subheader("📊 Data Overview")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(hist_df))
                with col2:
                    st.metric("Date Range", f"{hist_df['timestamp'].min().strftime('%Y-%m-%d')} to {hist_df['timestamp'].max().strftime('%Y-%m-%d')}")
                with col3:
                    st.metric("Avg AQI", f"{hist_df['aqi'].mean():.1f}")
                
                # AQI Time Series
                st.subheader("📈 AQI Over Time")
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=hist_df['timestamp'],
                    y=hist_df['aqi'],
                    mode='lines+markers',
                    name='AQI',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    title="Historical AQI Values",
                    xaxis_title="Time",
                    yaxis_title="AQI",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Weather correlations
                if 'temperature' in hist_df.columns and 'humidity' in hist_df.columns:
                    st.subheader("🌡️ Weather Correlations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Temperature vs AQI
                        fig_temp = px.scatter(
                            hist_df, x='temperature', y='aqi',
                            title="Temperature vs AQI",
                            labels={'temperature': 'Temperature (°C)', 'aqi': 'AQI'}
                        )
                        st.plotly_chart(fig_temp, use_container_width=True)
                    
                    with col2:
                        # Humidity vs AQI
                        fig_humidity = px.scatter(
                            hist_df, x='humidity', y='aqi',
                            title="Humidity vs AQI",
                            labels={'humidity': 'Humidity (%)', 'aqi': 'AQI'}
                        )
                        st.plotly_chart(fig_humidity, use_container_width=True)
                
                # Data table
                st.subheader("📋 Raw Data")
                display_df = hist_df[['timestamp', 'aqi', 'temperature', 'humidity', 'wind_speed']].copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                display_df.columns = ['Timestamp', 'AQI', 'Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)']
                
                st.dataframe(display_df, use_container_width=True)
                
            else:
                st.warning("No historical data available. Please run the feature pipeline to collect data.")
                
        except Exception as e:
            st.error(f"Error displaying historical data: {e}")
            logger.error(f"Error in historical_data_section: {e}")
    
    def model_info_section(self):
        """Display model information."""
        st.header("🤖 Model Information")
        
        try:
            # Get model performance
            performance = self.model_trainer.get_model_performance()
            
            if performance:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Model Performance")
                    for metric, value in performance.items():
                        if isinstance(value, float):
                            st.metric(metric, f"{value:.4f}")
                        else:
                            st.metric(metric, value)
                
                with col2:
                    st.subheader("🏆 Best Model")
                    best_model = self.model_trainer.get_best_model_name()
                    if best_model:
                        st.success(f"**{best_model}**")
                        st.info("This model achieved the best performance on the validation set.")
                    else:
                        st.warning("No best model identified.")
                
                # Feature importance
                st.subheader("🎯 Feature Importance")
                importance = self.model_trainer.get_feature_importance()
                
                if importance:
                    importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
                    importance_df = importance_df.sort_values('Importance', ascending=True)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance (Top 10)'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for the current model.")
                    
            else:
                st.warning("No model performance data available. Please train a model first.")
                
        except Exception as e:
            st.error(f"Error displaying model information: {e}")
            logger.error(f"Error in model_info_section: {e}")
    
    def get_aqi_category(self, aqi_value):
        """Get AQI category based on value."""
        if aqi_value <= 50:
            return "Good"
        elif aqi_value <= 100:
            return "Moderate"
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi_value <= 200:
            return "Unhealthy"
        elif aqi_value <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def generate_predictions(self, model, days=3):
        """Generate AQI predictions for the next N days."""
        try:
            # Get historical features to understand the expected structure
            features_df = self.feature_pipeline.get_features()
            if features_df.empty:
                logger.warning("No historical features available for prediction")
                return None
            
            # Get the feature columns (excluding target)
            feature_columns = [col for col in features_df.columns if col not in ['aqi', 'timestamp']]
            
            # Create future dates
            future_dates = []
            current_time = datetime.now()
            
            for i in range(days * 24):  # Hourly predictions
                future_time = current_time + timedelta(hours=i)
                future_dates.append(future_time)
            
            # Create features for future dates
            future_features = []
            for date in future_dates:
                # Start with basic time features
                features = {
                    'hour': date.hour,
                    'day': date.day,
                    'month': date.month,
                    'day_of_week': date.weekday(),
                    'is_weekend': 1 if date.weekday() >= 5 else 0,
                    'season': (date.month % 12 + 3) // 3
                }
                
                # Add cyclical features
                features['hour_sin'] = np.sin(2 * np.pi * date.hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * date.hour / 24)
                features['day_sin'] = np.sin(2 * np.pi * date.day / 31)
                features['day_cos'] = np.cos(2 * np.pi * date.day / 31)
                features['month_sin'] = np.sin(2 * np.pi * date.month / 12)
                features['month_cos'] = np.cos(2 * np.pi * date.month / 12)
                
                # Add weather features (use historical averages as reasonable estimates)
                if 'temperature' in feature_columns:
                    features['temperature'] = features_df['temperature'].mean()
                    features['temperature_squared'] = features['temperature'] ** 2
                    features['temperature_cubed'] = features['temperature'] ** 3
                
                if 'humidity' in feature_columns:
                    features['humidity'] = features_df['humidity'].mean()
                    features['humidity_squared'] = features['humidity'] ** 2
                
                if 'pressure' in feature_columns:
                    features['pressure'] = features_df['pressure'].mean()
                    features['pressure_normalized'] = (features['pressure'] - 1013.25) / 1013.25
                
                if 'wind_speed' in feature_columns:
                    features['wind_speed'] = features_df['wind_speed'].mean()
                    features['wind_speed_squared'] = features['wind_speed'] ** 2
                
                if 'wind_direction' in feature_columns:
                    features['wind_direction'] = features_df['wind_direction'].mean()
                    features['wind_direction_sin'] = np.sin(np.radians(features['wind_direction']))
                    features['wind_direction_cos'] = np.cos(np.radians(features['wind_direction']))
                
                if 'visibility' in feature_columns:
                    features['visibility'] = features_df['visibility'].mean()
                    features['visibility_normalized'] = features['visibility'] / 10000
                
                if 'clouds' in feature_columns:
                    features['clouds'] = features_df['clouds'].mean()
                    features['clouds_normalized'] = features['clouds'] / 100
                
                # Add pollutant features (use historical averages)
                pollutant_features = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
                for pollutant in pollutant_features:
                    if pollutant in feature_columns:
                        features[pollutant] = features_df[pollutant].mean()
                        features[f'{pollutant}_squared'] = features[pollutant] ** 2
                        features[f'{pollutant}_log'] = np.log1p(features[pollutant])
                
                # Add derived features
                if 'pollutant_sum' in feature_columns:
                    available_pollutants = [f for f in pollutant_features if f in features]
                    if available_pollutants:
                        features['pollutant_sum'] = sum(features[f] for f in available_pollutants)
                        features['pollutant_mean'] = features['pollutant_sum'] / len(available_pollutants)
                        features['pollutant_std'] = features_df['pollutant_std'].mean() if 'pollutant_std' in features_df.columns else 0
                        features['pollutant_max'] = max(features[f] for f in available_pollutants)
                
                # Add interaction features
                if 'temp_humidity_interaction' in feature_columns and 'temperature' in features and 'humidity' in features:
                    features['temp_humidity_interaction'] = features['temperature'] * features['humidity']
                
                if 'pm25_pm10_ratio' in feature_columns and 'pm25' in features and 'pm10' in features:
                    features['pm25_pm10_ratio'] = features['pm25'] / features['pm10'] if features['pm10'] > 0 else 0
                
                if 'wind_pm25_interaction' in feature_columns and 'wind_speed' in features and 'pm25' in features:
                    features['wind_pm25_interaction'] = features['wind_speed'] * features['pm25']
                
                # Add lag features (use historical averages)
                lag_features = ['temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3',
                              'humidity_lag_1', 'humidity_lag_2', 'humidity_lag_3',
                              'wind_speed_lag_1', 'wind_speed_lag_2', 'wind_speed_lag_3',
                              'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3',
                              'pm10_lag_1', 'pm10_lag_2', 'pm10_lag_3']
                
                for lag_feature in lag_features:
                    if lag_feature in feature_columns:
                        features[lag_feature] = features_df[lag_feature].mean()
                
                # Add change rate features (use historical averages)
                change_rate_features = ['temperature_change_rate', 'temperature_change_rate_pct',
                                      'humidity_change_rate', 'humidity_change_rate_pct',
                                      'wind_speed_change_rate', 'wind_speed_change_rate_pct',
                                      'pm25_change_rate', 'pm25_change_rate_pct',
                                      'pm10_change_rate', 'pm10_change_rate_pct']
                
                for change_feature in change_rate_features:
                    if change_feature in feature_columns:
                        features[change_feature] = features_df[change_feature].mean()
                
                # Add rolling statistics features (use historical averages)
                rolling_features = ['temperature_rolling_mean_24h', 'temperature_rolling_std_24h',
                                  'temperature_rolling_min_24h', 'temperature_rolling_max_24h',
                                  'humidity_rolling_mean_24h', 'humidity_rolling_std_24h',
                                  'humidity_rolling_min_24h', 'humidity_rolling_max_24h',
                                  'wind_speed_rolling_mean_24h', 'wind_speed_rolling_std_24h',
                                  'wind_speed_rolling_min_24h', 'wind_speed_rolling_max_24h',
                                  'pm25_rolling_mean_24h', 'pm25_rolling_std_24h',
                                  'pm25_rolling_min_24h', 'pm25_rolling_max_24h',
                                  'pm10_rolling_mean_24h', 'pm10_rolling_std_24h',
                                  'pm10_rolling_min_24h', 'pm10_rolling_max_24h',
                                  'temperature_rolling_mean_7d', 'temperature_rolling_std_7d',
                                  'humidity_rolling_mean_7d', 'humidity_rolling_std_7d',
                                  'wind_speed_rolling_mean_7d', 'wind_speed_rolling_std_7d',
                                  'pm25_rolling_mean_7d', 'pm25_rolling_std_7d',
                                  'pm10_rolling_mean_7d', 'pm10_rolling_std_7d']
                
                for rolling_feature in rolling_features:
                    if rolling_feature in feature_columns:
                        features[rolling_feature] = features_df[rolling_feature].mean()
                
                future_features.append(features)
            
            # Convert to DataFrame
            future_df = pd.DataFrame(future_features)
            
            # Ensure all required columns are present and in the right order
            missing_columns = [col for col in feature_columns if col not in future_df.columns]
            for col in missing_columns:
                future_df[col] = 0  # Fill missing columns with 0
            
            # Reorder columns to match training data
            future_df = future_df[feature_columns]
            
            # Make predictions
            predictions = model.predict(future_df)
            
            # Create prediction results
            results = []
            for i, (date, pred) in enumerate(zip(future_dates, predictions)):
                results.append({
                    'date': date,
                    'aqi': max(0, pred),  # AQI cannot be negative
                    'category': self.get_aqi_category(pred),
                    'confidence': 0.85 - (i * 0.02)  # Decreasing confidence over time
                })
            
            logger.info(f"Generated {len(results)} predictions successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return None
    
    def display_health_recommendations(self, predictions):
        """Display health recommendations based on predictions."""
        max_aqi = max(pred['aqi'] for pred in predictions)
        max_category = self.get_aqi_category(max_aqi)
        
        recommendations = {
            'Good': {
                'icon': '✅',
                'message': 'Air quality is satisfactory, and air pollution poses little or no risk.',
                'actions': ['Enjoy outdoor activities', 'Open windows for ventilation', 'Normal outdoor exercise']
            },
            'Moderate': {
                'icon': '⚠️',
                'message': 'Air quality is acceptable; however, some pollutants may be a concern for a small number of people.',
                'actions': ['Sensitive individuals should limit outdoor activities', 'Consider indoor exercise', 'Monitor symptoms']
            },
            'Unhealthy for Sensitive Groups': {
                'icon': '🔶',
                'message': 'Members of sensitive groups may experience health effects.',
                'actions': ['Sensitive groups should reduce outdoor activities', 'Keep windows closed', 'Use air purifiers']
            },
            'Unhealthy': {
                'icon': '🔴',
                'message': 'Everyone may begin to experience health effects.',
                'actions': ['Limit outdoor activities', 'Stay indoors with air conditioning', 'Wear masks if going outside']
            },
            'Very Unhealthy': {
                'icon': '🟣',
                'message': 'Health warnings of emergency conditions.',
                'actions': ['Avoid outdoor activities', 'Stay indoors', 'Use air purifiers', 'Consider evacuation']
            },
            'Hazardous': {
                'icon': '⚫',
                'message': 'Health alert: everyone may experience more serious health effects.',
                'actions': ['Stay indoors', 'Avoid all outdoor activities', 'Use air purifiers', 'Consider evacuation']
            }
        }
        
        rec = recommendations.get(max_category, recommendations['Good'])
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>{rec['icon']} {max_category} Air Quality Expected</h4>
            <p>{rec['message']}</p>
            <h5>Recommended Actions:</h5>
            <ul>
                {''.join([f'<li>{action}</li>' for action in rec['actions']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def display_fun_facts(self):
        """Display interesting facts about air quality."""
        st.subheader("🧠 Did You Know?")
        
        facts = [
            "🌱 Trees can remove up to 60% of air pollution in urban areas!",
            "🏃‍♂️ Exercise during high AQI can be 3x more harmful than during clean air",
            "🏠 Indoor air can be 2-5x more polluted than outdoor air",
            "🌍 The cleanest air on Earth is found in Tasmania, Australia",
            "🚗 A single car can emit 4.6 metric tons of CO2 per year",
            "🌿 Houseplants like spider plants can improve indoor air quality by 20%",
            "☀️ Sunny days often have worse air quality due to ozone formation",
            "🌊 Coastal areas typically have better air quality due to sea breezes"
        ]
        
        # Randomly select 3 facts
        import random
        selected_facts = random.sample(facts, 3)
        
        for fact in selected_facts:
            st.info(fact)
    
    def display_achievement_badges(self):
        """Display achievement badges for user engagement."""
        st.subheader("🏆 Your Air Quality Achievements")
        
        # Get user stats
        features_df = self.feature_pipeline.get_features()
        if not features_df.empty:
            total_records = len(features_df)
            avg_aqi = features_df['aqi'].mean()
            max_aqi = features_df['aqi'].max()
            min_aqi = features_df['aqi'].min()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if total_records >= 100:
                    st.success("📊 Data Collector\n100+ records collected!")
                elif total_records >= 50:
                    st.info("📈 Data Enthusiast\n50+ records collected!")
                else:
                    st.warning("📝 Beginner\nKeep collecting data!")
            
            with col2:
                if avg_aqi <= 30:
                    st.success("🌿 Clean Air Champion\nExcellent average AQI!")
                elif avg_aqi <= 60:
                    st.info("😊 Air Quality Aware\nGood average AQI!")
                else:
                    st.warning("⚠️ Air Quality Monitor\nMonitor air quality closely!")
            
            with col3:
                if max_aqi <= 50:
                    st.success("🛡️ Safety Guardian\nNever experienced unhealthy air!")
                elif max_aqi <= 100:
                    st.info("👀 Air Quality Watcher\nModerate air quality experienced!")
                else:
                    st.warning("🚨 Air Quality Alert\nUnhealthy air experienced!")
            
            with col4:
                if min_aqi <= 20:
                    st.success("🌅 Perfect Air Seeker\nExperienced pristine air!")
                elif min_aqi <= 40:
                    st.info("🌤️ Clean Air Explorer\nExperienced good air!")
                else:
                    st.warning("🌫️ Air Quality Learner\nLearning about air quality!")
        else:
            st.info("Start collecting data to earn achievements! 🚀")
    
    def display_trend_analysis(self):
        """Display trend analysis with insights."""
        st.subheader("📈 Trend Analysis & Insights")
        
        features_df = self.feature_pipeline.get_features()
        if not features_df.empty:
            hist_df = features_df.sort_values('timestamp').copy()
            
            # Calculate trends
            if len(hist_df) >= 2:
                recent_aqi = hist_df['aqi'].tail(10).mean()
                older_aqi = hist_df['aqi'].head(10).mean()
                trend = recent_aqi - older_aqi
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Recent AQI (Last 10)", f"{recent_aqi:.1f}")
                
                with col2:
                    st.metric("Older AQI (First 10)", f"{older_aqi:.1f}")
                
                with col3:
                    if trend < -5:
                        st.success(f"📉 Improving Trend\nAQI decreased by {abs(trend):.1f}")
                    elif trend > 5:
                        st.error(f"📈 Worsening Trend\nAQI increased by {trend:.1f}")
                    else:
                        st.info(f"➡️ Stable Trend\nAQI changed by {trend:.1f}")
                
                # Time-based insights
                st.subheader("⏰ Time-Based Insights")
                
                if 'hour' in hist_df.columns:
                    hourly_avg = hist_df.groupby('hour')['aqi'].mean()
                    worst_hour = hourly_avg.idxmax()
                    best_hour = hourly_avg.idxmin()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"🕐 Worst Time: {worst_hour}:00\nAverage AQI: {hourly_avg[worst_hour]:.1f}")
                    with col2:
                        st.success(f"🌅 Best Time: {best_hour}:00\nAverage AQI: {hourly_avg[best_hour]:.1f}")
                
                # Weather correlation insights
                if 'temperature' in hist_df.columns and 'humidity' in hist_df.columns:
                    st.subheader("🌡️ Weather Insights")
                    
                    temp_corr = hist_df['temperature'].corr(hist_df['aqi'])
                    humidity_corr = hist_df['humidity'].corr(hist_df['aqi'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if abs(temp_corr) > 0.3:
                            if temp_corr > 0:
                                st.warning(f"🌡️ Temperature Correlation: {temp_corr:.2f}\nHigher temps = Higher AQI")
                            else:
                                st.info(f"🌡️ Temperature Correlation: {temp_corr:.2f}\nLower temps = Higher AQI")
                        else:
                            st.info(f"🌡️ Temperature Correlation: {temp_corr:.2f}\nWeak correlation")
                    
                    with col2:
                        if abs(humidity_corr) > 0.3:
                            if humidity_corr > 0:
                                st.warning(f"💧 Humidity Correlation: {humidity_corr:.2f}\nHigher humidity = Higher AQI")
                            else:
                                st.info(f"💧 Humidity Correlation: {humidity_corr:.2f}\nLower humidity = Higher AQI")
                        else:
                            st.info(f"💧 Humidity Correlation: {humidity_corr:.2f}\nWeak correlation")
    
    def display_real_time_alerts(self):
        """Display real-time alerts for hazardous conditions."""
        st.subheader("🚨 Real-Time Air Quality Alerts")
        
        try:
            # Get current data
            current_data = self.feature_pipeline.collect_current_data()
            if current_data and 'air_pollution_openweather' in current_data:
                aqi = current_data['air_pollution_openweather'].get('aqi', 0)
                
                if aqi >= 300:
                    st.error("🚨 HAZARDOUS AIR QUALITY ALERT! 🚨\n\n"
                            "⚠️ EMERGENCY CONDITIONS\n"
                            "• Everyone should avoid all outdoor activities\n"
                            "• Stay indoors with air conditioning\n"
                            "• Use air purifiers if available\n"
                            "• Consider evacuation if conditions persist")
                elif aqi >= 200:
                    st.error("🟣 VERY UNHEALTHY AIR QUALITY ALERT!\n\n"
                            "🚨 HEALTH WARNINGS\n"
                            "• Avoid outdoor activities\n"
                            "• Stay indoors with air conditioning\n"
                            "• Use air purifiers\n"
                            "• Monitor health symptoms")
                elif aqi >= 150:
                    st.warning("🔴 UNHEALTHY AIR QUALITY ALERT!\n\n"
                              "⚠️ HEALTH EFFECTS POSSIBLE\n"
                              "• Limit outdoor activities\n"
                              "• Stay indoors with air conditioning\n"
                              "• Wear masks if going outside\n"
                              "• Monitor symptoms")
                elif aqi >= 100:
                    st.warning("🟠 UNHEALTHY FOR SENSITIVE GROUPS\n\n"
                              "👥 SENSITIVE INDIVIDUALS\n"
                              "• Sensitive groups should reduce outdoor activities\n"
                              "• Keep windows closed\n"
                              "• Use air purifiers\n"
                              "• Monitor symptoms")
                elif aqi >= 50:
                    st.info("🟡 MODERATE AIR QUALITY\n\n"
                            "⚠️ ACCEPTABLE BUT CONCERNING\n"
                            "• Air quality is acceptable\n"
                            "• Some pollutants may concern sensitive individuals\n"
                            "• Consider indoor exercise\n"
                            "• Monitor symptoms")
                else:
                    st.success("🟢 GOOD AIR QUALITY\n\n"
                              "✅ EXCELLENT CONDITIONS\n"
                              "• Air quality is satisfactory\n"
                              "• No health risks\n"
                              "• Enjoy outdoor activities\n"
                              "• Open windows for ventilation")
            else:
                st.info("🔍 Checking current air quality...")
                
        except Exception as e:
            st.warning("Unable to get current air quality data")
    
    def run(self):
        """Run the Streamlit application."""
        self.main_header()
        self.location_selector()
        
        # Real-time alerts at the top
        self.display_real_time_alerts()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Current AQI", "🔮 Predictions", "📈 History", "🎯 Fun & Insights"])
        
        with tab1:
            self.current_aqi_section()
        
        with tab2:
            self.predictions_section()
        
        with tab3:
            self.historical_data_section()
        
        with tab4:
            st.header("🎯 Fun & Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                self.display_fun_facts()
                self.display_achievement_badges()
            
            with col2:
                self.display_trend_analysis()

def main():
    """Main function to run the Streamlit app."""
    try:
        app = AQIPredictorApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main() 