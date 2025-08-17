"""
Logging utilities for AQI Predictor.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from .config import config


def setup_logger(
    name: str = "aqi_predictor",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path
        
    Returns:
        Configured logger
    """
    # Get configuration with defaults
    try:
        log_config = config.get('logging', {})
    except:
        log_config = {}
    
    level = level or log_config.get('level', 'INFO')
    log_file = log_file or log_config.get('file', 'logs/aqi_predictor.log')
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        max_size = log_config.get('max_size_mb', 10) * 1024 * 1024  # Convert to bytes
        backup_count = log_config.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "aqi_predictor") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default logger - only create if not already exists
if not logging.getLogger().handlers:
    logger = setup_logger()
else:
    logger = logging.getLogger("aqi_predictor") 