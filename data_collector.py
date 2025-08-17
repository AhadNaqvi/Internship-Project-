"""
Data collection module for AQI Predictor.
Fetches weather and air quality data from multiple APIs.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import time
import random
from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)

class DataCollector:
    """Collect weather and air quality data from various APIs."""
    
    def __init__(self):
        """Initialize the data collector."""
        self.location = config.get_location()
        self.apis = config.get_apis()
        
        # API keys from environment
        self.openweathermap_api_key = config.get('OPENWEATHERMAP_API_KEY', '')
        self.aqicn_api_key = config.get('AQICN_API_KEY', '')
        self.weatherapi_key = config.get('WEATHERAPI_API_KEY', '')
        
        # Base URLs
        self.openweathermap_base = self.apis['openweathermap']['base_url']
        self.aqicn_base = self.apis['aqicn']['base_url']
        self.weatherapi_base = self.apis['weatherapi']['base_url']
    
    def _make_request(self, url: str, params: Dict[str, Any] = None, retries: int = 3) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: Request URL
            params: Query parameters
            retries: Number of retry attempts
            
        Returns:
            Response data
        """
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(random.uniform(1, 3))  # Random delay
                else:
                    raise
    
    def get_openweathermap_data(self) -> Dict[str, Any]:
        """
        Get current weather data from OpenWeatherMap.
        
        Returns:
            Weather data dictionary
        """
        try:
            url = f"{self.openweathermap_base}/weather"
            params = {
                'lat': self.location['latitude'],
                'lon': self.location['longitude'],
                'appid': self.openweathermap_api_key,
                'units': 'metric'
            }
            
            data = self._make_request(url, params)
            
            weather_data = {
                'timestamp': datetime.now(),
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon'],
                'visibility': data.get('visibility', 0),
                'clouds': data['clouds']['all'],
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']),
                'sunset': datetime.fromtimestamp(data['sys']['sunset'])
            }
            
            logger.info(f"Successfully collected OpenWeatherMap data for {self.location['city']}")
            return weather_data
            
        except Exception as e:
            logger.error(f"Error collecting OpenWeatherMap data: {e}")
            raise
    
    def get_openweathermap_air_pollution(self) -> Dict[str, Any]:
        """
        Get air pollution data from OpenWeatherMap.
        
        Returns:
            Air pollution data dictionary
        """
        try:
            url = f"{self.openweathermap_base}/air_pollution"
            params = {
                'lat': self.location['latitude'],
                'lon': self.location['longitude'],
                'appid': self.openweathermap_api_key
            }
            
            data = self._make_request(url, params)
            
            # Calculate AQI from components
            components = data['list'][0]['components']
            aqi = data['list'][0]['main']['aqi']
            
            # Convert AQI to standard scale (1-500)
            aqi_standard = {
                1: 50,
                2: 100,
                3: 150,
                4: 200,
                5: 300
            }.get(aqi, 100)
            
            air_pollution_data = {
                'timestamp': datetime.fromtimestamp(data['list'][0]['dt']),
                'aqi': aqi_standard,
                'pm25': components.get('pm2_5', 0),
                'pm10': components.get('pm10', 0),
                'o3': components.get('o3', 0),
                'no2': components.get('no2', 0),
                'so2': components.get('so2', 0),
                'co': components.get('co', 0),
                'nh3': components.get('nh3', 0)
            }
            
            logger.info(f"Successfully collected OpenWeatherMap air pollution data for {self.location['city']}")
            return air_pollution_data
            
        except Exception as e:
            logger.error(f"Error collecting OpenWeatherMap air pollution data: {e}")
            raise
    
    def get_aqicn_data(self) -> Dict[str, Any]:
        """
        Get air quality data from AQICN.
        
        Returns:
            AQICN data dictionary
        """
        try:
            url = f"{self.aqicn_base}/@{self.location['latitude']};{self.location['longitude']}/"
            params = {
                'token': self.aqicn_api_key
            }
            
            data = self._make_request(url, params)
            
            if data.get('status') == 'error':
                raise Exception(f"AQICN API error: {data.get('data', 'Unknown error')}")
            
            aqicn_data = {
                'timestamp': datetime.now(),
                'aqi': data['data']['aqi'],
                'pm25': data['data']['iaqi'].get('pm25', {}).get('v', 0),
                'pm10': data['data']['iaqi'].get('pm10', {}).get('v', 0),
                'o3': data['data']['iaqi'].get('o3', {}).get('v', 0),
                'no2': data['data']['iaqi'].get('no2', {}).get('v', 0),
                'so2': data['data']['iaqi'].get('so2', {}).get('v', 0),
                'co': data['data']['iaqi'].get('co', {}).get('v', 0),
                't': data['data']['iaqi'].get('t', {}).get('v', 0),
                'h': data['data']['iaqi'].get('h', {}).get('v', 0),
                'p': data['data']['iaqi'].get('p', {}).get('v', 0),
                'w': data['data']['iaqi'].get('w', {}).get('v', 0),
                'wg': data['data']['iaqi'].get('wg', {}).get('v', 0)
            }
            
            logger.info(f"Successfully collected AQICN data for {self.location['city']}")
            return aqicn_data
            
        except Exception as e:
            logger.error(f"Error collecting AQICN data: {e}")
            raise
    
    def get_weather_forecast(self) -> Dict[str, Any]:
        """
        Get weather forecast from OpenWeatherMap.
        
        Returns:
            Weather forecast data
        """
        try:
            url = f"{self.openweathermap_base}/forecast"
            params = {
                'lat': self.location['latitude'],
                'lon': self.location['longitude'],
                'appid': self.openweathermap_api_key,
                'units': 'metric'
            }
            
            data = self._make_request(url, params)
            
            forecast_data = {
                'timestamp': datetime.now(),
                'forecast': []
            }
            
            for item in data['list']:
                forecast_item = {
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                    'wind_direction': item['wind'].get('deg', 0),
                    'description': item['weather'][0]['description'],
                    'icon': item['weather'][0]['icon'],
                    'clouds': item['clouds']['all'],
                    'pop': item.get('pop', 0)  # Probability of precipitation
                }
                forecast_data['forecast'].append(forecast_item)
            
            logger.info(f"Successfully collected weather forecast for {self.location['city']}")
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error collecting weather forecast: {e}")
            raise
    
    def collect_all_data(self) -> Dict[str, Any]:
        """
        Collect all available data from multiple sources.

        Returns:
            Combined data from all sources
        """
        try:
            # Collect current weather and air quality data
            weather_data = self.get_openweathermap_data()
            air_pollution = self.get_openweathermap_air_pollution()

            # Try to get AQICN data, but don't fail if unavailable
            aqicn_data = {}
            try:
                aqicn_data = self.get_aqicn_data()
                logger.info("Successfully collected AQICN data")
            except Exception as e:
                logger.warning(f"Could not collect AQICN data: {e}")
                # Create placeholder AQICN data using OpenWeatherMap air pollution data
                aqicn_data = {
                    'timestamp': air_pollution['timestamp'],
                    'aqi': air_pollution['aqi'],
                    'pm25': air_pollution['pm25'],
                    'pm10': air_pollution['pm10'],
                    'o3': air_pollution['o3'],
                    'no2': air_pollution['no2'],
                    'so2': air_pollution['so2'],
                    'co': air_pollution['co']
                }

            # Combine data
            combined_data = {
                'timestamp': datetime.now(),
                'location': self.location,
                'weather': weather_data,
                'air_pollution_openweather': air_pollution,
                'air_quality_aqicn': aqicn_data,
                'source': 'data_collector'
            }

            logger.info("Successfully collected all data")
            return combined_data

        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            raise
    
    def collect_historical_data(self, start_date: datetime, end_date: datetime) -> list:
        """
        Collect historical data for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of historical data points
        """
        historical_data = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                # Simulate historical data collection
                # In a real implementation, you would call historical APIs
                weather_data = {
                    'timestamp': current_date,
                    'temperature': 20 + random.uniform(-5, 5),
                    'humidity': 50 + random.uniform(-20, 20),
                    'pressure': 1013 + random.uniform(-10, 10),
                    'wind_speed': random.uniform(0, 10),
                    'wind_direction': random.uniform(0, 360),
                    'description': 'Clear sky',
                    'icon': '01d'
                }
                
                air_pollution_data = {
                    'timestamp': current_date,
                    'aqi': 50 + random.uniform(0, 100),
                    'pm25': random.uniform(5, 25),
                    'pm10': random.uniform(10, 50),
                    'o3': random.uniform(20, 80),
                    'no2': random.uniform(10, 40),
                    'so2': random.uniform(2, 15),
                    'co': random.uniform(200, 800)
                }
                
                historical_data.append({
                    'timestamp': current_date,
                    'location': self.location,
                    'weather': weather_data,
                    'air_pollution_openweather': air_pollution_data,
                    'air_quality_aqicn': air_pollution_data,
                    'source': 'historical'
                })
                
                current_date += timedelta(hours=1)
                
            except Exception as e:
                logger.error(f"Error collecting historical data for {current_date}: {e}")
                current_date += timedelta(hours=1)
        
        logger.info(f"Collected {len(historical_data)} historical data points")
        return historical_data 