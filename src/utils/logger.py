"""
Logger Utility Module

This module provides logging functionality for the vulnerability detection system.
It sets up loggers with proper formatting and output configuration.
"""

import os
import logging
import logging.config
import yaml
from typing import Optional

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for the specified module.
    
    Args:
        name: Name of the module (usually __name__)
        
    Returns:
        Configured logger
    """
    # Create a default logger if the configuration hasn't been loaded
    logger = logging.getLogger(name)
    
    # Don't add a handler if the logger already has handlers
    if logger.handlers:
        return logger
    
    # Check if the logging.yaml exists
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'logging.yaml')
    
    if os.path.exists(config_path):
        # Load the logging configuration
        try:
            with open(config_path, 'rt') as f:
                config = yaml.safe_load(f.read())
            
            # Create logs directory if it doesn't exist
            logs_dir = config.get('handlers', {}).get('file_handler', {}).get('filename', '')
            if logs_dir:
                os.makedirs(os.path.dirname(logs_dir), exist_ok=True)
            
            error_logs_dir = config.get('handlers', {}).get('error_file_handler', {}).get('filename', '')
            if error_logs_dir:
                os.makedirs(os.path.dirname(error_logs_dir), exist_ok=True)
            
            # Apply the configuration
            logging.config.dictConfig(config)
            
            # Get the configured logger
            logger = logging.getLogger(name)
        except Exception as e:
            # If there's an error loading the configuration, set up a basic logger
            _setup_basic_logger(logger)
            logger.error(f"Error loading logging configuration: {e}")
    else:
        # Set up a basic logger if the configuration file doesn't exist
        _setup_basic_logger(logger)
        logger.warning(f"Logging configuration file not found at {config_path}. Using basic configuration.")
    
    return logger

def _setup_basic_logger(logger: logging.Logger) -> None:
    """
    Set up a basic logger with console output.
    
    Args:
        logger: Logger to configure
    """
    # Set the log level
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a file handler
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'vulnerability_detection.log'))
    file_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def initialize_logging(config_path: Optional[str] = None) -> None:
    """
    Initialize logging for the application.
    
    Args:
        config_path: Path to the logging configuration file (optional)
    """
    # Use default path if not provided
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'logging.yaml')
    
    # Check if the config file exists
    if os.path.exists(config_path):
        # Load the logging configuration
        try:
            with open(config_path, 'rt') as f:
                config = yaml.safe_load(f.read())
            
            # Create logs directory if it doesn't exist
            logs_dir = config.get('handlers', {}).get('file_handler', {}).get('filename', '')
            if logs_dir:
                os.makedirs(os.path.dirname(logs_dir), exist_ok=True)
            
            error_logs_dir = config.get('handlers', {}).get('error_file_handler', {}).get('filename', '')
            if error_logs_dir:
                os.makedirs(os.path.dirname(error_logs_dir), exist_ok=True)
            
            # Apply the configuration
            logging.config.dictConfig(config)
            
            # Get the root logger
            logger = logging.getLogger()
            logger.info("Logging initialized successfully from configuration file.")
        except Exception as e:
            # If there's an error loading the configuration, set up a basic logger
            logger = logging.getLogger()
            _setup_basic_logger(logger)
            logger.error(f"Error loading logging configuration: {e}")
    else:
        # Set up a basic logger if the configuration file doesn't exist
        logger = logging.getLogger()
        _setup_basic_logger(logger)
        logger.warning(f"Logging configuration file not found at {config_path}. Using basic configuration.")