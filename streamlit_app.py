import streamlit as st
import os
import time
from datetime import datetime
import nltk

# Import core modules
from text_preprocessing import IndonesianTextPreprocessor, FuzzyMatcher
from model_training import ChatbotModel
from config import config

# Page Config
st.set_page_config(
    page_title="HLO Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Configuration
env = os.getenv('FLASK_ENV', 'production')
conf = config['default']

# --- Helper Functions ---

@st.cache_resource
def setup_nltk():
    """Download necessary NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

@st.cache_resource
def load_and_prepare_model():
    """Load model or train if not exists"""
    
    # 1. Setup NLTK
    setup_nltk()

    model_path = conf.MODEL_PATH
    vectorizer_path = conf.VECTORIZER_PATH
    encoder_path = conf.LABEL_ENCODER_PATH
    dataset_path = conf.DATASET_PATH
    
    # Check if model exists
    if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(encoder_path):
        try:
            model = ChatbotModel()
            model.load_model(model_path, vectorizer_path, encoder_path)
            
            # Load fuzzy matcher
            fuzzy = FuzzyMatcher(dataset_path)
            
            return model, fuzzy, "Model loaded successfully"
        except Exception as e:
            st.warning(f"Failed to load existing model: {e}. Retraining...")
    
    # If not exists or failed to load, Train it
    status_text = "Training model from scratch..."
    
    try:
        from dataset_preprocessing import DatasetPreprocessor
        
        processor = DatasetPreprocessor(dataset_path)
        processed_data = processor.preprocess_dataset(stem=True, remove_stopwords=False)
        
        if not processed_data:
            return None, None, "Dataset is empty!"
            
        X = [item['preprocessed'] for item in processed_data]
        y = [item['intent'] for item in processed_data]
        
        model = ChatbotModel()
        results = model.train(X, y)
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path, vectorizer_path, encoder_path)
        
        # Load fuzzy
        fuzzy = FuzzyMatcher(dataset_path)
        
        return model, fuzzy, f"Model trained! Accuracy: {results['accuracy']:.2f}"
        
    except Exception as e:
        return None, None, f"Error training model: {e}"

def get_response_from_knowledge(intent, user_message, dataset_path):
    import json
    import random
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        for intent_data in dataset['intents']:
            if intent_data['tag'] == intent:
                response = random.choice(intent_data['responses'])
                
                # Simple enhancement logic (simplified from app.py)
                if intent in ['harga_kaos', 'produk_kaos', 'merchandise_lain']:
                     # For now just return the response to avoid complex dependency on knowledgeBase.json
                     # unless we load it too. Let's keep it simple first.
                     pass
                return response
        return "Maaf, saya tidak mengerti."
    except Exception:
        return "Maaf terjadi kesalahan sistem."

# --- Main App ---

st.title("🤖 HLO Chatbot")
st.markdown("Asisten Virtual HLO (Machine Learning Based)")

# Sidebar
with st.sidebar:
    st.header("Status System")
    
    with st.spinner("Preparing AI Model..."):
        model, fuzzy, status = load_and_prepare_model()
        
    if model:
        st.success(status)
    else:
        st.error(status)
        st.stop()
        
    # Reset Chat
    if st.button("Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize Preprocessor
preprocessor = IndonesianTextPreprocessor()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ketik pesan Anda di sini..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            # Preprocess
            processed_msg = preprocessor.preprocess(prompt, stem=True, remove_stopwords=False)
            
            # Predict
            intent, confidence = model.predict(processed_msg)
            
            # Fuzzy Fallback
            if confidence < 0.4 and fuzzy:
                fuzzy_result = fuzzy.find_best_match(prompt, threshold=0.6)
                if fuzzy_result:
                    _, f_intent, f_score = fuzzy_result
                    if f_score > confidence:
                        intent = f_intent
                        confidence = f_score
            
            # Get Response
            response = get_response_from_knowledge(intent, prompt, conf.DATASET_PATH)
            
            # Simulate typing
            time.sleep(0.5)
            
            st.markdown(response)
            
            # Log debug info in expander
            with st.expander("Debug Info"):
                st.text(f"Intent: {intent}")
                st.text(f"Confidence: {confidence:.2f}")

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
