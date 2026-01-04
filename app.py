from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
import pickle
import numpy as np
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps

from text_preprocessing import IndonesianTextPreprocessor, FuzzyMatcher
from model_training import ChatbotModel
from config import config

app = Flask(__name__)

env = os.getenv('FLASK_ENV', 'production')
app.config.from_object(config[env])

CORS(app, resources={
    r"/api/*": {
        "origins": app.config['CORS_ORIGINS'],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[app.config['RATE_LIMIT_DEFAULT']],
    storage_uri="memory://"
)

if not os.path.exists(app.config['LOG_DIR']):
    os.makedirs(app.config['LOG_DIR'])

file_handler = RotatingFileHandler(
    app.config['LOG_FILE'],
    maxBytes=app.config['LOG_MAX_BYTES'],
    backupCount=app.config['LOG_BACKUP_COUNT']
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(getattr(logging, app.config['LOG_LEVEL']))
app.logger.addHandler(file_handler)
app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))
app.logger.info('HLO Chatbot Backend startup')

preprocessor = IndonesianTextPreprocessor()
fuzzy_matcher = None
chatbot_model = None

MODEL_PATH = app.config['MODEL_PATH']
VECTORIZER_PATH = app.config['VECTORIZER_PATH']
LABEL_ENCODER_PATH = app.config['LABEL_ENCODER_PATH']
DATASET_PATH = app.config['DATASET_PATH']
KNOWLEDGE_PATH = app.config['KNOWLEDGE_PATH']

try:
    with open(KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    app.logger.info('Knowledge base loaded successfully')
except Exception as e:
    app.logger.error(f'Failed to load knowledge base: {e}')
    knowledge_base = {"products": {"kaos": [], "merchandise": []}}

chat_sessions = {}

def validate_json(required_fields):

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Content-Type must be application/json'
                }), 400
            
            data = request.json
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

def load_model():
    global chatbot_model, fuzzy_matcher
    
    if not all([
        os.path.exists(MODEL_PATH),
        os.path.exists(VECTORIZER_PATH),
        os.path.exists(LABEL_ENCODER_PATH)
    ]):
        app.logger.warning("Model files not found")
        return False
    
    try:
        chatbot_model = ChatbotModel()
        chatbot_model.load_model(MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH)
        
        fuzzy_matcher = FuzzyMatcher(DATASET_PATH)
        
        app.logger.info("Model and fuzzy matcher loaded successfully")
        return True
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        return False

def get_response_from_knowledge(intent, user_message):

    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        for intent_data in dataset['intents']:
            if intent_data['tag'] == intent:
                import random
                response = random.choice(intent_data['responses'])
                
                if intent == 'harga_kaos':
                    response = enhance_harga_kaos(response)
                elif intent == 'merchandise_lain':
                    response = enhance_merchandise_lain(response)
                elif intent == 'produk_kaos':
                    response = enhance_produk_kaos(response)
                
                return response
        
        return "Maaf, saya belum memahami pertanyaan Anda. Bisa dijelaskan lebih detail?"
    
    except Exception as e:
        app.logger.error(f"Error getting response: {e}")
        return "Maaf, terjadi kesalahan sistem. Silakan coba lagi."

def enhance_harga_kaos(base_response):

    try:
        products = knowledge_base['products']['kaos']
        
        detail = "\n\nDETAIL HARGA KAOS HLO:\n"
        for product in products:
            name = product['nama'].replace('Premium ', '')
            retail = f"Rp {product['harga']['retail']:,}"
            detail += f"\n- {name}: {retail}"
        
        detail += "\n\nBeli grosir lebih hemat!"
        return base_response + detail
    except Exception as e:
        app.logger.error(f"Error enhancing harga_kaos: {e}")
        return base_response

def enhance_merchandise_lain(base_response):

    try:
        merch = knowledge_base['products']['merchandise']
        
        detail = "\n\nMERCHANDISE HLO:\n"
        
        affordable = [m for m in merch if m['harga']['retail'] < 25000]
        premium = [m for m in merch if m['harga']['retail'] >= 25000]
        
        detail += "\nAffordable (< 25rb):\n"
        for item in affordable[:3]:
            detail += f"- {item['nama']}: Rp {item['harga']['retail']:,}\n"
        
        detail += "\nPremium Items:\n"
        for item in premium[:3]:
            detail += f"- {item['nama']}: Rp {item['harga']['retail']:,}\n"
        
        return base_response + detail
    except Exception as e:
        app.logger.error(f"Error enhancing merchandise: {e}")
        return base_response

def enhance_produk_kaos(base_response):

    try:
        products = knowledge_base['products']['kaos']
        
        detail = "\n\nKOLEKSI KAOS HLO:\n"
        for product in products:
            name = product['nama']
            price = f"Rp {product['harga']['retail']:,}"
            detail += f"\n- {name}\n  Harga: {price}\n  Bahan: {product['bahan']}\n"
        
        return base_response + detail
    except Exception as e:
        app.logger.error(f"Error enhancing produk_kaos: {e}")
        return base_response

@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        'success': False,
        'error': 'Bad Request',
        'message': str(e)
    }), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Not Found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f'Internal error: {e}')
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': 'An internal error occurred'
    }), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({
        'success': False,
        'error': 'File Too Large',
        'message': 'The uploaded file is too large (max 16MB)'
    }), 413

@app.route('/api/health', methods=['GET'])
def health_check():

    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': chatbot_model is not None,
        'version': '1.0.0'
    })

@app.route('/api/chat', methods=['POST'])
@limiter.limit(app.config['RATE_LIMIT_CHAT'])
@validate_json(['message'])
def chat():

    
    if chatbot_model is None:
        app.logger.warning('Chat request received but model not loaded')
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'message': 'Please train the model first'
        }), 503
    
    data = request.json
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    
    if len(user_message) > app.config['MESSAGE_MAX_LENGTH']:
        return jsonify({
            'success': False,
            'error': 'Message too long',
            'message': 'Message must be less than 500 characters'
        }), 400
    
    try:
        processed_message = preprocessor.preprocess(
            user_message,
            stem=True,
            remove_stopwords=False
        )
        
        intent, confidence = chatbot_model.predict(processed_message)
        
        if confidence < 0.4 and fuzzy_matcher:
            fuzzy_result = fuzzy_matcher.find_best_match(user_message, threshold=0.6)
            if fuzzy_result:
                fuzzy_pattern, fuzzy_intent, fuzzy_score = fuzzy_result
                if fuzzy_score > confidence:
                    intent = fuzzy_intent
                    confidence = fuzzy_score
                    app.logger.info(f'Fuzzy match used: {fuzzy_intent} (score: {fuzzy_score:.2f})')
        
        response = get_response_from_knowledge(intent, user_message)
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_entry = {
            'user_message': user_message,
            'bot_response': response,
            'intent': intent,
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat()
        }
        
        chat_sessions[session_id].append(chat_entry)
        
        if len(chat_sessions[session_id]) > app.config['SESSION_MAX_MESSAGES']:
            chat_sessions[session_id] = chat_sessions[session_id][-app.config['SESSION_MAX_MESSAGES']:]
        
        app.logger.info(f'Chat processed - Session: {session_id}, Intent: {intent}, Confidence: {confidence:.2f}')
        
        return jsonify({
            'success': True,
            'response': response,
            'intent': intent,
            'confidence': float(confidence),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f'Error processing chat: {e}')
        return jsonify({
            'success': False,
            'error': 'Processing error',
            'message': 'Error processing your message'
        }), 500

@app.route('/api/train', methods=['POST'])
@limiter.limit(app.config['RATE_LIMIT_TRAIN'])
def train_model():

    
    try:
        global chatbot_model
        
        app.logger.info('Model training started')
        
        chatbot_model = ChatbotModel()
        
        from dataset_preprocessing import DatasetPreprocessor
        
        dataset_processor = DatasetPreprocessor(DATASET_PATH)
        processed_data = dataset_processor.preprocess_dataset(
            stem=True,
            remove_stopwords=False
        )
        
        if not processed_data:
            return jsonify({
                'success': False,
                'error': 'No training data',
                'message': 'Dataset is empty or invalid'
            }), 400
        
        X = [item['preprocessed'] for item in processed_data]
        y = [item['intent'] for item in processed_data]
        
        results = chatbot_model.train(X, y)
        
        os.makedirs('models', exist_ok=True)
        chatbot_model.save_model(MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH)
        
        app.logger.info(f'Model trained successfully - Accuracy: {results["accuracy"]:.4f}')
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'accuracy': float(results['accuracy']),
            'cv_mean': float(results['cv_mean']),
            'cv_std': float(results['cv_std']),
            'samples': len(X),
            'intents': results['n_intents']
        })
        
    except Exception as e:
        app.logger.error(f'Error training model: {e}')
        return jsonify({
            'success': False,
            'error': 'Training error',
            'message': str(e)
        }), 500

@app.route('/api/products', methods=['GET'])
def get_products():

    
    category = request.args.get('category', 'all')
    
    try:
        if category == 'kaos':
            products = knowledge_base['products']['kaos']
        elif category == 'merchandise':
            products = knowledge_base['products']['merchandise']
        else:
            products = {
                'kaos': knowledge_base['products']['kaos'],
                'merchandise': knowledge_base['products']['merchandise']
            }
        
        return jsonify({
            'success': True,
            'products': products,
            'count': len(products) if isinstance(products, list) else sum(len(v) for v in products.values())
        })
        
    except Exception as e:
        app.logger.error(f'Error fetching products: {e}')
        return jsonify({
            'success': False,
            'error': 'Fetch error',
            'message': 'Error fetching products'
        }), 500

@app.route('/api/intents', methods=['GET'])
def get_intents():

    
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        intents = [
            {
                'tag': intent['tag'],
                'patterns_count': len(intent['patterns']),
                'responses_count': len(intent['responses'])
            }
            for intent in dataset['intents']
        ]
        
        return jsonify({
            'success': True,
            'intents': intents,
            'total': len(intents)
        })
        
    except Exception as e:
        app.logger.error(f'Error fetching intents: {e}')
        return jsonify({
            'success': False,
            'error': 'Fetch error',
            'message': 'Error fetching intents'
        }), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):

    
    limit = request.args.get('limit', 50, type=int)
    
    history = chat_sessions.get(session_id, [])
    
    if limit > 0:
        history = history[-limit:]
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'history': history,
        'count': len(history)
    })

@app.route('/api/history/<session_id>', methods=['DELETE'])
def clear_chat_history(session_id):

    
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        app.logger.info(f'Chat history cleared for session: {session_id}')
    
    return jsonify({
        'success': True,
        'message': 'Chat history cleared',
        'session_id': session_id
    })

@app.route('/api/stats', methods=['GET'])
def get_statistics():

    
    try:
        model_exists = os.path.exists(MODEL_PATH)
        
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        total_intents = len(dataset['intents'])
        total_patterns = sum(len(intent['patterns']) for intent in dataset['intents'])
        
        total_kaos = len(knowledge_base['products']['kaos'])
        total_merch = len(knowledge_base['products']['merchandise'])
        
        total_sessions = len(chat_sessions)
        total_messages = sum(len(history) for history in chat_sessions.values())
        
        return jsonify({
            'success': True,
            'stats': {
                'model': {
                    'trained': model_exists,
                    'loaded': chatbot_model is not None
                },
                'dataset': {
                    'total_intents': total_intents,
                    'total_patterns': total_patterns
                },
                'products': {
                    'total_kaos': total_kaos,
                    'total_merchandise': total_merch,
                    'total': total_kaos + total_merch
                },
                'chat': {
                    'total_sessions': total_sessions,
                    'total_messages': total_messages,
                    'active_sessions': len([s for s in chat_sessions if len(chat_sessions[s]) > 0])
                }
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f'Error fetching statistics: {e}')
        return jsonify({
            'success': False,
            'error': 'Stats error',
            'message': 'Error fetching statistics'
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("HLO CHATBOT BACKEND SERVER")
    print("="*60)
    
    model_loaded = load_model()
    
    if not model_loaded:
        print("\nModel not found!")
        print("Tip: Send POST request to /api/train to train the model first")
    else:
        print("\nModel loaded successfully!")
    
    print("\nServer starting...")
    print(" API Endpoints:")
    print("   Health:")
    print("   - GET  /api/health")
    print("\n   Chat:")
    print("   - POST /api/chat")
    print("   - GET  /api/history/<session_id>")
    print("   - DELETE /api/history/<session_id>")
    print("\n   Model:")
    print("   - POST /api/train")
    print("\n   Data:")
    print("   - GET  /api/products")
    print("   - GET  /api/intents")
    print("   - GET  /api/stats")
    print("\n" + "="*60 + "\n")
    
    app.run(
        debug=app.config['DEBUG'],
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
