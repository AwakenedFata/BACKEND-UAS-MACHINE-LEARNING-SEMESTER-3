import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import logging

logger = logging.getLogger(__name__)

class ChatbotModel:

    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        self.label_encoder = LabelEncoder()
        
        self.nb_model = MultinomialNB(alpha=1.0)
        self.svm_model = SVC(kernel='linear', probability=True, C=1.0)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('nb', self.nb_model),
                ('svm', self.svm_model),
                ('rf', self.rf_model)
            ],
            voting='soft',
            weights=[1, 2, 1]
        )
        
        self.is_trained = False
    
    def train(self, X, y, test_size=0.2, random_state=42):

        
        print("TRAINING HLO CHATBOT MODEL")
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        print("\nVectorizing text with TF-IDF...")
        X_vectorized = self.vectorizer.fit_transform(X)
        
        print(f"Feature dimension: {X_vectorized.shape[1]}")
        print(f"Total samples: {len(X)}")
        print(f"Total intents: {len(set(y))}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        print(f"\nTraining samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        
        logger.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
        
        print("\nTraining individual models...")
        
        print("  1. Naive Bayes...")
        self.nb_model.fit(X_train, y_train)
        nb_score = self.nb_model.score(X_test, y_test)
        print(f"     Accuracy: {nb_score:.4f}")
        
        print("  2. SVM...")
        self.svm_model.fit(X_train, y_train)
        svm_score = self.svm_model.score(X_test, y_test)
        print(f"     Accuracy: {svm_score:.4f}")
        
        print("  3. Random Forest...")
        self.rf_model.fit(X_train, y_train)
        rf_score  = self.rf_model.score(X_test, y_test)
        print(f"     Accuracy: {rf_score:.4f}")
        
        print("\nTraining Ensemble Model...")
        self.ensemble_model.fit(X_train, y_train)
        ensemble_score = self.ensemble_model.score(X_test, y_test)
        print(f"  Ensemble Accuracy: {ensemble_score:.4f}")
        
        print("\nCross-validation (5-fold)...")
        cv_scores = cross_val_score(
            self.ensemble_model, X_vectorized, y_encoded, cv=5
        )
        print(f"  CV Scores: {cv_scores}")
        print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        y_pred = self.ensemble_model.predict(X_test)
        
        print("\nCLASSIFICATION REPORT:")

        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        print(report)
        
        cm = confusion_matrix(y_test, y_pred)
        
        self.is_trained = True
        
        logger.info(f"Training completed - Ensemble accuracy: {ensemble_score:.4f}")
        
        print("\nTraining completed successfully!")
        
        results = {
            'accuracy': ensemble_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'nb_accuracy': nb_score,
            'svm_accuracy': svm_score,
            'rf_accuracy': rf_score,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0],
            'n_intents': len(set(y))
        }
        
        return results
    
    def predict(self, text):

        
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        X = self.vectorizer.transform([text])
        
        y_pred = self.ensemble_model.predict(X)[0]
        
        probas = self.ensemble_model.predict_proba(X)[0]
        confidence = np.max(probas)
        
        intent = self.label_encoder.inverse_transform([y_pred])[0]
        
        return intent, confidence
    
    def predict_with_alternatives(self, text, top_n=3):

        
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        X = self.vectorizer.transform([text])
        
        probas = self.ensemble_model.predict_proba(X)[0]
        
        top_indices = np.argsort(probas)[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            intent = self.label_encoder.classes_[idx]
            confidence = probas[idx]
            results.append((intent, confidence))
        
        return results
    
    def save_model(self, model_path, vectorizer_path, encoder_path):

        
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"Model saved to: {model_path}")
    
    def load_model(self, model_path, vectorizer_path, encoder_path):

        
        with open(model_path, 'rb') as f:
            self.ensemble_model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.is_trained = True
        
        logger.info(f"Model loaded from: {model_path}")

if __name__ == "__main__":
    import os
    from dataset_preprocessing import DatasetPreprocessor
    from config import config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Loading and preprocessing dataset...")
    # Initialize preprocessor with dataset path from config
    # config is a dict of classes, so we access default config properties
    conf = config['default']
    
    dataset_proc = DatasetPreprocessor(conf.DATASET_PATH)
    
    # Preprocess patterns
    dataset_proc.preprocess_dataset()
    
    # Get dataframe
    df = dataset_proc.get_dataframe()
    
    if df is not None:
        X = df['preprocessed'].tolist()
        y = df['intent'].tolist()
        
        # Initialize and train model
        model = ChatbotModel()
        results = model.train(X, y)
        
        # Save model
        print(f"\nSaving model to {conf.MODEL_PATH}...")
        model.save_model(
            conf.MODEL_PATH,
            conf.VECTORIZER_PATH,
            conf.LABEL_ENCODER_PATH
        )
        print("Done!")
    else:
        print("Error: Dataset is empty or could not be loaded")
