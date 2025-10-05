"""
Blooming prediction system configuration
"""

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True

# Weather data configuration
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# Default coordinates (Jefferson City, MO)
DEFAULT_LATITUDE = 38.6275
DEFAULT_LONGITUDE = -92.5666

# Model configuration
MODEL_FILE = "models/blooming_model.pkl"
CACHE_DIR = ".cache"

# Prediction configuration
DEFAULT_DAYS_AHEAD = 14
MAX_DAYS_AHEAD = 30
MIN_DAYS_AHEAD = 1

# Feature configuration
BLOOM_SCORE_THRESHOLD_HIGH = 0.7
BLOOM_SCORE_THRESHOLD_MEDIUM = 0.5

# Ideal conditions for Blooming
IDEAL_TEMPERATURE_MIN = 15
IDEAL_TEMPERATURE_MAX = 25
IDEAL_HUMIDITY_MIN = 40
IDEAL_HUMIDITY_MAX = 70
IDEAL_PRECIPITATION_MAX = 5

# Visualization configuration
CHART_WIDTH = 15
CHART_HEIGHT = 10
CHART_DPI = 300

# Cache configuration
CACHE_EXPIRE_AFTER = -1  # Permanent cache
RETRY_ATTEMPTS = 5
RETRY_BACKOFF_FACTOR = 0.2

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Validation configuration
MIN_LATITUDE = -90
MAX_LATITUDE = 90
MIN_LONGITUDE = -180
MAX_LONGITUDE = 180

# Performance configuration
MAX_SAMPLES_FOR_TRAINING = 10000
CROSS_VALIDATION_FOLDS = 5
RANDOM_STATE = 42

# File configuration
DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
SCRIPTS_DIR = "scripts"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"
