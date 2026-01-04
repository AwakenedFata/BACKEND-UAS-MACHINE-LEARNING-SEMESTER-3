import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    JSON_AS_ASCII = False
    
    MODEL_PATH = 'models/chatbot_model.pkl'
    VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
    LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
    DATASET_PATH = 'datasets.json'
    KNOWLEDGE_PATH = 'knowledgeBase.json'
    
    LOG_DIR = 'logs'
    LOG_FILE = 'logs/app.log'
    LOG_MAX_BYTES = 10240000
    LOG_BACKUP_COUNT = 10
    
    RATE_LIMIT_DEFAULT = os.getenv('RATE_LIMIT_DEFAULT', '200 per day, 50 per hour')
    RATE_LIMIT_CHAT = os.getenv('RATE_LIMIT_CHAT', '30 per minute')
    RATE_LIMIT_TRAIN = os.getenv('RATE_LIMIT_TRAIN', '5 per hour')
    
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://localhost:3000').split(',')
    
    SESSION_MAX_MESSAGES = 50
    MESSAGE_MIN_LENGTH = 1
    MESSAGE_MAX_LENGTH = 500

class DevelopmentConfig(Config):

    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):

    DEBUG = False
    LOG_LEVEL = 'INFO'
    

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}
