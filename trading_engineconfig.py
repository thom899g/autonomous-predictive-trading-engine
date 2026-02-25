"""
Configuration management for trading engine.
Centralizes all environment variables, constants, and settings.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Configuration container for trading engine."""
    
    # API Credentials
    EXCHANGE_API_KEY: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    EXCHANGE_API_SECRET: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))
    EXCHANGE_API_PASSPHRASE: Optional[str] = os.getenv("EXCHANGE_API_PASSPHRASE")
    
    # Firebase Configuration
    FIREBASE_CREDENTIALS_PATH: Path = field(
        default_factory=lambda: Path(os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json"))
    )
    FIREBASE_PROJECT_ID: str = field(default_factory=lambda: os.getenv("FIREBASE_PROJECT_ID", ""))
    
    # Trading Parameters
    INITIAL_CAPITAL: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", 10000.0)))
    RISK_PER_TRADE: float = field(default_factory=lambda: float(os.getenv("RISK_PER_TRADE", 0.02)))  # 2% per trade
    MAX_POSITION_SIZE: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", 0.1)))  # 10% max
    
    # Model Parameters
    MODEL_SAVE_PATH: Path = field(default_factory=lambda: Path(os.getenv("MODEL_SAVE_PATH", "./models")))
    TRAINING_WINDOW_DAYS: int = field(default_factory=lambda: int(os.getenv("TRAINING_WINDOW_DAYS", 365)))
    PREDICTION_HORIZON_HOURS: int = field(default_factory=lambda: int(os.getenv("PREDICTION_HORIZON_HOURS", 24)))
    
    # Data Collection
    MARKET_DATA_INTERVAL: str = field(default_factory=lambda: os.getenv("MARKET_DATA_INTERVAL", "1h"))
    SYMBOLS_TO_TRACK: list = field(default_factory=lambda: os.getenv("SYMBOLS_TO_TRACK", "BTC/USDT,ETH/USDT").split(","))
    
    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_FILE_PATH: Path = field(default_factory=lambda: Path(os.getenv("LOG_FILE_PATH", "./logs")))
    
    def validate(self) -> bool:
        """Validate configuration values."""
        validation_errors = []
        
        if not self.EXCHANGE_API_KEY:
            validation_errors.append("EXCHANGE_API_KEY is required")
        if not self.EXCHANGE_API_SECRET:
            validation_errors.append("EXCHANGE_API_SECRET is required")
        if not self.FIREBASE_PROJECT_ID:
            validation_errors.append("FIREBASE_PROJECT_ID is required")
        if not self.FIREBASE_CREDENTIALS_PATH.exists():
            validation_errors.append(f"Firebase credentials not found at {self.FIREBASE_CREDENTIALS_PATH}")
        
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {', '.join(validation_errors)}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            key: value if not isinstance(value, Path) else str(value)
            for key, value in self.__dict__.items()
            if not key.startswith("_") and not key in ["EXCHANGE_API_SECRET", "EXCHANGE_API_KEY"]
        }

# Global configuration instance
config = TradingConfig()