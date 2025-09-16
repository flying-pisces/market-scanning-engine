"""
Step 1: Technical Indicators - Only SMA, RSI, MACD
Simple, fast, and verifiable calculations
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average
    
    Args:
        prices: Series of closing prices
        window: Period for SMA (default 20)
    
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=window, min_periods=window).mean()


def calculate_ema(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        prices: Series of closing prices
        window: Period for EMA
    
    Returns:
        Series of EMA values
    """
    return prices.ewm(span=window, adjust=False).mean()


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        prices: Series of closing prices
        window: Period for RSI (default 14)
    
    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices: pd.Series, 
                   fast: int = 12, 
                   slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


class TechnicalIndicators:
    """Container for all technical indicators for a symbol"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Calculate all indicators from OHLCV data
        
        Args:
            data: DataFrame with OHLCV columns
        """
        self.data = data
        self.close = data['Close']
        
        # Calculate indicators
        self.sma_20 = calculate_sma(self.close, 20)
        self.sma_50 = calculate_sma(self.close, 50)
        self.rsi = calculate_rsi(self.close, 14)
        self.macd, self.macd_signal, self.macd_histogram = calculate_macd(self.close)
        
        # Get latest values for signal generation
        self.latest = self._get_latest_values()
    
    def _get_latest_values(self) -> dict:
        """Get the latest indicator values for signal generation"""
        if len(self.close) < 50:  # Not enough data
            return None
        
        return {
            'price': float(self.close.iloc[-1]),
            'sma_20': float(self.sma_20.iloc[-1]) if not pd.isna(self.sma_20.iloc[-1]) else None,
            'sma_50': float(self.sma_50.iloc[-1]) if not pd.isna(self.sma_50.iloc[-1]) else None,
            'rsi': float(self.rsi.iloc[-1]) if not pd.isna(self.rsi.iloc[-1]) else None,
            'macd': float(self.macd.iloc[-1]) if not pd.isna(self.macd.iloc[-1]) else None,
            'macd_signal': float(self.macd_signal.iloc[-1]) if not pd.isna(self.macd_signal.iloc[-1]) else None,
            'macd_histogram': float(self.macd_histogram.iloc[-1]) if not pd.isna(self.macd_histogram.iloc[-1]) else None,
        }
    
    def get_signal_features(self) -> dict:
        """
        Get features for signal generation
        
        Returns:
            Dictionary of signal features
        """
        if self.latest is None:
            return None
        
        features = {
            'price_above_sma20': self.latest['price'] > self.latest['sma_20'] if self.latest['sma_20'] else None,
            'price_above_sma50': self.latest['price'] > self.latest['sma_50'] if self.latest['sma_50'] else None,
            'sma20_above_sma50': self.latest['sma_20'] > self.latest['sma_50'] if self.latest['sma_20'] and self.latest['sma_50'] else None,
            'rsi_oversold': self.latest['rsi'] < 30 if self.latest['rsi'] else None,
            'rsi_overbought': self.latest['rsi'] > 70 if self.latest['rsi'] else None,
            'macd_bullish': self.latest['macd'] > self.latest['macd_signal'] if self.latest['macd'] and self.latest['macd_signal'] else None,
            'macd_histogram_positive': self.latest['macd_histogram'] > 0 if self.latest['macd_histogram'] else None,
        }
        
        # Add momentum features
        if len(self.close) >= 5:
            features['momentum_5d'] = float((self.close.iloc[-1] - self.close.iloc[-5]) / self.close.iloc[-5] * 100)
        
        if len(self.close) >= 20:
            features['momentum_20d'] = float((self.close.iloc[-1] - self.close.iloc[-20]) / self.close.iloc[-20] * 100)
        
        return features


def quick_signals(data: pd.DataFrame) -> str:
    """
    Generate quick buy/sell/hold signal from data
    
    Args:
        data: OHLCV DataFrame
    
    Returns:
        'BUY', 'SELL', or 'HOLD'
    """
    indicators = TechnicalIndicators(data)
    features = indicators.get_signal_features()
    
    if features is None:
        return 'HOLD'
    
    buy_signals = 0
    sell_signals = 0
    
    # Count bullish signals
    if features.get('price_above_sma20') and features.get('price_above_sma50'):
        buy_signals += 1
    if features.get('sma20_above_sma50'):
        buy_signals += 1
    if features.get('rsi_oversold'):
        buy_signals += 2  # Strong signal
    if features.get('macd_bullish'):
        buy_signals += 1
    if features.get('momentum_5d', 0) > 2:
        buy_signals += 1
    
    # Count bearish signals
    if not features.get('price_above_sma20') and not features.get('price_above_sma50'):
        sell_signals += 1
    if not features.get('sma20_above_sma50'):
        sell_signals += 1
    if features.get('rsi_overbought'):
        sell_signals += 2  # Strong signal
    if not features.get('macd_bullish'):
        sell_signals += 1
    if features.get('momentum_5d', 0) < -2:
        sell_signals += 1
    
    # Decision logic
    if buy_signals >= 3 and buy_signals > sell_signals:
        return 'BUY'
    elif sell_signals >= 3 and sell_signals > buy_signals:
        return 'SELL'
    else:
        return 'HOLD'