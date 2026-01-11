"""Constants used across the trading lab system."""

# Data directories
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
ARTIFACTS_DIR = "data/artifacts"

# Price data subdirectories
PRICES_DIR = "prices/yfinance"
NEWS_DIR = "news/newsapi"
SOCIAL_DIR = "social/reddit"
FUNDAMENTALS_DIR = "fundamentals"
MACRO_DIR = "macro"

# Processed data subdirectories
UNIFIED_DIR = "unified"
FEATURES_DIR = "features"

# Artifacts subdirectories
MODELS_DIR = "models"
BACKTESTS_DIR = "backtests"
REPORTS_DIR = "reports"

# Default model parameters
DEFAULT_TRAIN_WINDOW_YEARS = 2
DEFAULT_TEST_WINDOW_MONTHS = 3
DEFAULT_STEP_MONTHS = 1

# Default trading parameters
DEFAULT_MAX_POSITION_PER_ASSET = 0.1
DEFAULT_MAX_GROSS_EXPOSURE = 1.0
DEFAULT_TRANSACTION_COST_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_MAX_DRAWDOWN_THRESHOLD = 0.2

# Feature engineering defaults
DEFAULT_FEATURE_LOOKBACK_DAYS = 60
DEFAULT_MIN_PRICE_CHANGE_THRESHOLD = 0.0005  # 0.05%

# Label thresholds
DEFAULT_RETURN_THRESHOLD = 0.0005  # 0.05% for classification
DEFAULT_COST_BUFFER = 0.001  # 0.1% buffer for transaction cost-aware labeling

# Date formats
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

