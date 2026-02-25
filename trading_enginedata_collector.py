"""
Market data collector with error handling and Firebase integration.
Handles real-time and historical data collection from exchanges.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import ccxt
from ccxt import Exchange, NetworkError, ExchangeError, RequestTimeout

from .config import config
from .firebase_client import FirebaseClient

logger = logging.getLogger(__name__)

class MarketDataCollector:
    """Collects and manages market data with retry logic and error handling."""
    
    def __init__(self, exchange_id: str = "binance"):
        """
        Initialize market data collector.
        
        Args:
            exchange_id: Exchange identifier (default: binance)
        """
        self.exchange_id = exchange_id
        self.exchange: Optional[Exchange] = None
        self.firebase = FirebaseClient()
        self.initialize_exchange()
        
        # Initialize data cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.last_update_time: Dict[str, datetime] = {}
        
        logger.info(f"Initialized MarketDataCollector for {exchange_id}")
    
    def initialize_exchange(self) -> None:
        """Initialize exchange connection with error handling."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': config.EXCHANGE_API_KEY,
                'secret': config.EXCHANGE_API_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustFor