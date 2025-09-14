# Apponomics Configuration File

# Application Settings
APP_NAME = "Apponomics"
APP_VERSION = "1.0.0"
DEBUG = False

# Data Paths
DATA_DIR = "data"
MODELS_DIR = "models"
CONFIG_DIR = "config"

# Database Settings
DATABASE_PATH = "data/apponomics.db"

# ML Model Settings
MODEL_CONFIDENCE_THRESHOLD = 0.7
MAX_APPS_PER_USER = 50

# Tier Classification Thresholds
SPENDING_TIERS = {
    "premium_threshold": 4.0,
    "standard_threshold": 1.5,
    "basic_threshold": 0.0
}

GEOGRAPHIC_TIERS = {
    "tier1_threshold": 0.8,
    "tier2_threshold": 0.3,
    "tier3_threshold": 0.0
}

LIFESTYLE_CATEGORIES = {
    "professional_threshold": 0.6,
    "entertainment_threshold": 0.2,
    "basic_threshold": 0.0
}

# App Categories
NEUTRAL_APPS = {
    'food_delivery': ['zomato', 'swiggy', 'blinkit', 'zepto'],
    'transportation': ['ola', 'uber', 'rapido'],
    'payments': ['paytm', 'phonepe', 'google_pay', 'amazon_pay'],
    'ecommerce': ['flipkart', 'amazon', 'myntra'],
    'travel': ['irctc', 'makemytrip', 'goibibo'],
    'social_media': ['instagram', 'facebook', 'whatsapp', 'telegram'],
    'entertainment': ['youtube', 'netflix', 'spotify']
}

DISCRIMINATOR_APPS = {
    'tier_a_premium': [
        'cred', 'indmoney', 'zerodha', 'jupiter', 'groww',
        'airbnb', 'urban_company', 'nykaa_luxe', 'linkedin_premium',
        'apple_music', 'taj_hotels', 'binance', 'wazirx'
    ],
    'tier_b_mainstream': [
        'meesho', 'ajio', 'tata_neu', 'byju', 'unacademy',
        'sony_liv', 'magicbricks', 'apna', 'naukri',
        'bookmyshow', 'practo', 'policybazaar'
    ],
    'tier_c_budget': [
        'ludo_king', 'sharechat', 'moj', 'kreditbee', 'cashbean',
        'winzo', 'bharatpe_merchant', 'uc_browser', 'snack_video',
        'likee', 'mx_takatak', 'josh', 'chingari'
    ]
}

# Behavioral Weights
NEUTRAL_BEHAVIOR_WEIGHTS = {
    'food_delivery': {
        'frequency_weight': 0.05,
        'aov_weight': 0.1,
        'total_spend_weight': 0.08
    },
    'transportation': {
        'frequency_weight': 0.05,
        'aov_weight': 0.08,
        'total_spend_weight': 0.06
    },
    'payments': {
        'frequency_weight': 0.02,
        'aov_weight': 0.15,
        'total_spend_weight': 0.12
    },
    'ecommerce': {
        'frequency_weight': 0.05,
        'aov_weight': 0.1,
        'total_spend_weight': 0.08
    }
}

DISCRIMINATOR_WEIGHTS = {
    'tier_a_premium': {'spending': 1.0, 'geographic': 0.8, 'lifestyle': 0.8},
    'tier_b_mainstream': {'spending': 0.3, 'geographic': 0.4, 'lifestyle': 0.5},
    'tier_c_budget': {'spending': -0.4, 'geographic': -0.5, 'lifestyle': -0.3}
}
