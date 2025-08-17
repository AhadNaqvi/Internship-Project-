"""
Configuration management utilities for AQI Predictor.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for AQI Predictor."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.locations = self._load_locations()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise Exception(f"Error loading config from {self.config_path}: {e}")
    
    def _load_locations(self) -> Dict[str, Any]:
        """Load locations database from YAML file."""
        try:
            locations_path = "config/locations.yaml"
            with open(locations_path, 'r') as file:
                locations = yaml.safe_load(file)
            return locations.get('locations', {})
        except Exception as e:
            print(f"Warning: Could not load locations database: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key, checking environment variables first."""
        # First check environment variables
        env_value = os.environ.get(key)
        if env_value is not None:
            return env_value
        
        # Then check config file
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_location(self) -> Dict[str, Any]:
        """Get current location configuration."""
        return self.get('location', {})
    
    def get_apis(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get('apis', {})
    
    def get_features(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get('features', {})
    
    def get_models(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('models', {})
    
    def get_training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    def get_available_countries(self) -> List[Dict[str, str]]:
        """Get list of available countries with codes and names."""
        countries = []
        for code, data in self.locations.items():
            countries.append({
                'code': code,
                'name': data.get('name', code)
            })
        return sorted(countries, key=lambda x: x['name'])
    
    def get_available_cities(self, country_code: str) -> List[Dict[str, Any]]:
        """Get list of available cities for a country."""
        if country_code not in self.locations:
            return []
        
        cities = []
        country_data = self.locations[country_code]
        for city_name, city_data in country_data.get('cities', {}).items():
            cities.append({
                'name': city_name,
                'latitude': city_data.get('latitude'),
                'longitude': city_data.get('longitude'),
                'timezone': city_data.get('timezone')
            })
        return sorted(cities, key=lambda x: x['name'])
    
    def get_city_info(self, country_code: str, city_name: str) -> Optional[Dict[str, Any]]:
        """Get specific city information."""
        if country_code not in self.locations:
            return None
        
        cities = self.locations[country_code].get('cities', {})
        if city_name not in cities:
            return None
        
        city_data = cities[city_name]
        return {
            'name': city_name,
            'country_code': country_code,
            'country_name': self.locations[country_code].get('name', country_code),
            'latitude': city_data.get('latitude'),
            'longitude': city_data.get('longitude'),
            'timezone': city_data.get('timezone')
        }
    
    def update_location(self, country_code: str, city_name: str) -> bool:
        """Update the current location in configuration."""
        city_info = self.get_city_info(country_code, city_name)
        if not city_info:
            return False
        
        # Update the location in config
        self.config['location'] = {
            'city': city_name,
            'country': country_code,
            'latitude': city_info['latitude'],
            'longitude': city_info['longitude'],
            'timezone': city_info['timezone']
        }
        
        # Save updated config
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Warning: Could not save updated config: {e}")
            return False

# Global config instance
config = Config() 