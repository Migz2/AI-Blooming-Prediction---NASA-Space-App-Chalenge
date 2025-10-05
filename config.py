"""
Configurações do sistema de previsão de floração
"""

# Configurações da API
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True

# Configurações de dados meteorológicos
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# Coordenadas padrão (Jefferson City, MO)
DEFAULT_LATITUDE = 38.6275
DEFAULT_LONGITUDE = -92.5666

# Configurações do modelo
MODEL_FILE = "models/blooming_model.pkl"
CACHE_DIR = ".cache"

# Configurações de previsão
DEFAULT_DAYS_AHEAD = 14
MAX_DAYS_AHEAD = 30
MIN_DAYS_AHEAD = 1

# Configurações de features
BLOOM_SCORE_THRESHOLD_HIGH = 0.7
BLOOM_SCORE_THRESHOLD_MEDIUM = 0.5

# Condições ideais para floração
IDEAL_TEMPERATURE_MIN = 15
IDEAL_TEMPERATURE_MAX = 25
IDEAL_HUMIDITY_MIN = 40
IDEAL_HUMIDITY_MAX = 70
IDEAL_PRECIPITATION_MAX = 5

# Configurações de visualização
CHART_WIDTH = 15
CHART_HEIGHT = 10
CHART_DPI = 300

# Configurações de cache
CACHE_EXPIRE_AFTER = -1  # Cache permanente
RETRY_ATTEMPTS = 5
RETRY_BACKOFF_FACTOR = 0.2

# Configurações de logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configurações de validação
MIN_LATITUDE = -90
MAX_LATITUDE = 90
MIN_LONGITUDE = -180
MAX_LONGITUDE = 180

# Configurações de performance
MAX_SAMPLES_FOR_TRAINING = 10000
CROSS_VALIDATION_FOLDS = 5
RANDOM_STATE = 42

# Configurações de arquivos
DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
SCRIPTS_DIR = "scripts"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"
