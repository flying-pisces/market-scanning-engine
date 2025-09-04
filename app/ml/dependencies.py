"""
ML Dependencies with graceful fallbacks
Handles missing ML libraries gracefully for testing
"""

import warnings
from typing import Any, Optional

# Track what's available
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CVXPY_AVAILABLE = False
SCIPY_AVAILABLE = False
SKLEARN_AVAILABLE = False

# Try importing XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None

# Try importing LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None

# Try importing CVXPY
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    cp = None

# Try importing SciPy
try:
    import scipy
    import scipy.stats
    import scipy.optimize
    SCIPY_AVAILABLE = True
except ImportError:
    scipy = None

# Try importing scikit-learn
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    sklearn = None
    RandomForestClassifier = None
    RandomForestRegressor = None
    StandardScaler = None
    train_test_split = None

# Mock classes for missing dependencies
class MockModel:
    """Mock ML model for when dependencies are missing"""
    
    def __init__(self, *args, **kwargs):
        self.is_fitted = False
        self.feature_names = []
    
    def fit(self, X, y, **kwargs):
        self.is_fitted = True
        return self
    
    def predict(self, X, **kwargs):
        if hasattr(X, 'shape'):
            return [0.5] * X.shape[0]
        return [0.5] * len(X)
    
    def predict_proba(self, X, **kwargs):
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        return [[0.5, 0.5] for _ in range(n_samples)]
    
    @property
    def feature_importances_(self):
        return [0.1] * len(self.feature_names) if self.feature_names else [0.1, 0.2, 0.3]

class MockScaler:
    """Mock scaler for missing sklearn"""
    
    def __init__(self):
        self.is_fitted = False
    
    def fit(self, X):
        self.is_fitted = True
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        return X

# Provide fallbacks
def get_xgb_classifier(*args, **kwargs):
    if XGBOOST_AVAILABLE:
        return xgb.XGBClassifier(*args, **kwargs)
    else:
        warnings.warn("XGBoost not available, using mock model")
        return MockModel()

def get_xgb_regressor(*args, **kwargs):
    if XGBOOST_AVAILABLE:
        return xgb.XGBRegressor(*args, **kwargs)
    else:
        warnings.warn("XGBoost not available, using mock model")
        return MockModel()

def get_lgb_classifier(*args, **kwargs):
    if LIGHTGBM_AVAILABLE:
        return lgb.LGBMClassifier(*args, **kwargs)
    else:
        warnings.warn("LightGBM not available, using mock model")
        return MockModel()

def get_random_forest_classifier(*args, **kwargs):
    if SKLEARN_AVAILABLE:
        return RandomForestClassifier(*args, **kwargs)
    else:
        warnings.warn("Scikit-learn not available, using mock model")
        return MockModel()

def get_scaler():
    if SKLEARN_AVAILABLE:
        return StandardScaler()
    else:
        warnings.warn("Scikit-learn not available, using mock scaler")
        return MockScaler()

def get_train_test_split():
    if SKLEARN_AVAILABLE:
        return train_test_split
    else:
        def mock_split(X, y, test_size=0.2, random_state=None):
            split_idx = int(len(X) * (1 - test_size))
            return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
        return mock_split

# Optimization fallbacks
def solve_optimization_problem(*args, **kwargs):
    """Solve optimization problem with CVXPY or fallback"""
    if CVXPY_AVAILABLE:
        # Real CVXPY solving logic would go here
        pass
    else:
        warnings.warn("CVXPY not available, using simple fallback optimization")
        # Return a simple equal-weight solution
        n_assets = kwargs.get('n_assets', 3)
        return {f'asset_{i}': 1.0/n_assets for i in range(n_assets)}

# Statistical functions fallbacks
def get_norm_ppf():
    """Get normal distribution percentile point function"""
    if SCIPY_AVAILABLE:
        return scipy.stats.norm.ppf
    else:
        warnings.warn("SciPy not available, using simple normal approximation")
        def mock_ppf(q):
            # Very simple approximation for standard normal
            if q <= 0.01:
                return -2.33
            elif q <= 0.05:
                return -1.645
            elif q <= 0.5:
                return 0
            elif q <= 0.95:
                return 1.645
            else:
                return 2.33
        return mock_ppf

def check_ml_dependencies():
    """Check which ML dependencies are available"""
    return {
        'xgboost': XGBOOST_AVAILABLE,
        'lightgbm': LIGHTGBM_AVAILABLE,
        'cvxpy': CVXPY_AVAILABLE,
        'scipy': SCIPY_AVAILABLE,
        'sklearn': SKLEARN_AVAILABLE
    }

def get_dependency_status():
    """Get detailed dependency status"""
    status = check_ml_dependencies()
    available_count = sum(status.values())
    total_count = len(status)
    
    return {
        'status': status,
        'available': available_count,
        'total': total_count,
        'percentage': (available_count / total_count) * 100,
        'ready_for_production': available_count >= 3  # Need most dependencies
    }