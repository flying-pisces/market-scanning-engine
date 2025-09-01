"""
Custom exceptions for the Market Scanning Engine
"""

from datetime import datetime
from typing import Optional, Any, Dict


class APIException(Exception):
    """Base API exception"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)


class ValidationError(APIException):
    """Validation error exception"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class NotFoundError(APIException):
    """Resource not found exception"""
    
    def __init__(self, resource: str, identifier: str):
        message = f"{resource} with ID '{identifier}' not found"
        super().__init__(message, status_code=404)


class RiskScoringError(APIException):
    """Risk scoring calculation error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class SignalGenerationError(APIException):
    """Signal generation error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)